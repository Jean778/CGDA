"""Auditable CGDA evaluation primitives. No project data are read on import."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

SEEDS = [42, 2024, 7, 123, 2025]
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def check_disjoint(**sets):
    keys=list(sets)
    for i,a in enumerate(keys):
        assert len(sets[a])==len(np.unique(sets[a])),f'duplicate indices in {a}'
        for b in keys[i+1:]:
            assert not np.intersect1d(sets[a],sets[b]).size,f'overlap: {a}, {b}'

def stratified_parts(indices, y, fractions, seed):
    """Partition within observed classes, preserving input identity and minimum counts."""
    rng=np.random.RandomState(seed); out=[[] for _ in fractions]
    for c in np.unique(y[indices]):
        z=indices[y[indices]==c].copy(); rng.shuffle(z)
        counts=np.floor(np.array(fractions[:-1])*len(z)).astype(int)
        # Tiny classes: reserve evaluation samples rather than fabricate class coverage.
        if len(z)>=len(fractions):
            counts=np.maximum(counts,1)
            while counts.sum()>=len(z): counts[np.argmax(counts)]-=1
        cur=0
        for i,n in enumerate(counts): out[i].extend(z[cur:cur+n]); cur+=n
        out[-1].extend(z[cur:])
    return tuple(np.array(sorted(z),dtype=int) for z in out)

def nested_calibration(pool, target, y, fraction):
    # Pool is already randomized by the seed. A separate deterministic per-class ordering
    # is supplied by caller through pool; never examine feature values or model results.
    ans=[]
    for c in np.unique(y[target]):
        z=pool[y[pool]==c]
        n=min(len(z),max(1,int(fraction*np.sum(y[target]==c))))
        ans.extend(z[:n])
    return np.array(ans,dtype=int)

def ece(probs,y,classes,bins=10):
    conf=probs.max(1); correct=np.asarray(classes)[probs.argmax(1)]==y
    # Includes confidence exactly one in final bin.
    ids=np.minimum((conf*bins).astype(int),bins-1)
    return float(sum(np.mean(ids==k)*abs(correct[ids==k].mean()-conf[ids==k].mean())
                     for k in range(bins) if np.any(ids==k)))

def classification_metrics(y,p,probs=None,classes=None):
    result={'accuracy':accuracy_score(y,p),
            'macro_f1':f1_score(y,p,labels=np.unique(y),average='macro',zero_division=0)}
    if probs is not None: result['ece']=ece(probs,y,classes)
    for c in np.unique(y):
        result[f'f1_class_{c}']=f1_score(y,p,labels=[c],average=None,zero_division=0)[0]
    return result

def oscr(y,p,d,known):
    """Exact CCR against unknown false acceptance, with ties grouped; low d is known."""
    y=np.asarray(y); p=np.asarray(p); d=np.asarray(d); known=np.asarray(known,bool)
    assert known.any() and (~known).any()
    order=np.argsort(d,kind='stable'); ds=d[order]
    ends=np.r_[np.flatnonzero(np.diff(ds)!=0),len(ds)-1]
    correct=(known & (y==p))[order]
    ccr=np.r_[0,np.cumsum(correct)[ends]/known.sum()]
    fpr=np.r_[0,np.cumsum((~known)[order])[ends]/(~known).sum()]
    thr=np.r_[np.nextafter(ds[0],-np.inf),ds[ends]]
    curve=pd.DataFrame({'threshold':thr,'unknown_false_acceptance':fpr,'correct_classification_rate':ccr})
    return float(np.trapezoid(ccr,fpr)),curve

def rejection_metrics(y,p,d,known,threshold):
    a=d<=threshold; k=np.asarray(known,bool)
    return {'known_acceptance':float(a[k].mean()),'unknown_rejection':float((~a[~k]).mean()),
            'known_correct_acceptance':float((a[k] & (p[k]==y[k])).mean()),
            'unknown_false_acceptance':float(a[~k].mean())}

def binary_metrics(y,d,threshold):
    pred=(d>threshold).astype(int)
    return {'auroc':roc_auc_score(y,d),'average_precision':average_precision_score(y,d),
            'event_f1':f1_score(y,pred,zero_division=0),
            'false_alarm_rate':pred[y==0].mean(), 'event_sensitivity':pred[y==1].mean(),
            'event_miss_rate':1-pred[y==1].mean(),'accuracy':accuracy_score(y,pred)}

def quantile_threshold(d,q):
    assert len(d)>0 and np.isfinite(d).all()
    return float(np.quantile(d,q,method='higher'))

class Network(nn.Module):
    def __init__(self,dim,nclasses,small=False):
        super().__init__(); widths=[32,16] if small else [128,128,64]
        layers=[]; prev=dim
        for w in widths: layers.extend([nn.Linear(prev,w),nn.ReLU()]); prev=w
        self.fe=nn.Sequential(*layers); self.cls=nn.Linear(prev,nclasses)
    def forward(self,x): return self.cls(self.fe(x))

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lam): ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.lam*g,None

def train_network(xs,ys,xc,yc,mode,seed,classes,epochs=50,steps=None,small=False):
    """Equal optimizer updates and full batch size for all neural comparators."""
    torch.manual_seed(seed); rng=np.random.RandomState(seed)
    classes=np.asarray(classes); lut={int(c):i for i,c in enumerate(classes)}
    ys=np.array([lut[int(v)] for v in ys]); yc=np.array([lut[int(v)] for v in yc])
    model=Network(xs.shape[1],len(classes),small)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005 if small else .001)
    if mode=='dann':
        domain=nn.Linear(16 if small else 64,1)
        optimizer=torch.optim.Adam(list(model.parameters())+list(domain.parameters()),lr=.001)
    weights=np.ones(len(classes),np.float32)
    if not small:
        original={1:1.,2:1.,3:1.,4:1.5,5:1.,6:2.}
        weights=np.array([original.get(int(c),1.) for c in classes],np.float32)
    wt=torch.from_numpy(weights)
    use_cal=mode in ('cal_sup','cg_sup','cg_shuffle')
    xb=np.concatenate([xs,xc]) if use_cal and len(xc) else xs
    yb=np.r_[ys,yc] if use_cal and len(xc) else ys
    centers={c:xc[yc==c].mean(0) for c in np.unique(yc)}
    if mode=='cg_shuffle' and len(centers)>1:
        keys=sorted(centers); centers={c:centers[keys[(i+1)%len(keys)]] for i,c in enumerate(keys)}
    batch=128; steps=steps or math.ceil((len(xs)+len(xc)+len(classes)*60)/batch)
    for epoch in range(epochs):
        xa,ya=xb,yb
        if mode in ('cg_mix','cg_sup','cg_shuffle') and centers:
            xx=[]; yy=[]
            for c,center in centers.items():
                count=40 if small else 60
                ids=rng.choice(np.flatnonzero(yb==c),count,replace=True)
                lam=rng.beta(.5,.5,(count,1)).astype(np.float32)
                xx.append((1-lam)*xb[ids]+lam*center); yy.append(np.full(count,c))
            xa=np.concatenate([xb,*xx]); ya=np.concatenate([yb,*yy])
        xt=torch.from_numpy(np.asarray(xa,np.float32)); yt=torch.from_numpy(ya).long()
        order=np.resize(rng.permutation(len(xt)),steps*batch)
        for i in range(steps):
            ids=order[i*batch:(i+1)*batch]; x=xt[ids]; y=yt[ids]
            optimizer.zero_grad(set_to_none=True)
            if mode=='mixup_std':
                j=rng.permutation(len(x)); lam=float(rng.beta(.5,.5))
                logits=model(lam*x+(1-lam)*x[j])
                loss=lam*nn.functional.cross_entropy(logits,y,weight=wt)+(1-lam)*nn.functional.cross_entropy(logits,y[j],weight=wt)
            elif mode=='dann':
                h=model.fe(x); logits=model.cls(h)
                loss=nn.functional.cross_entropy(logits,y,weight=wt)
                if len(xc):
                    target=torch.from_numpy(xc[rng.randint(len(xc),size=batch)])
                    ds=domain(GRL.apply(h,.3)); dt=domain(GRL.apply(model.fe(target),.3))
                    loss=loss+.5*(nn.functional.binary_cross_entropy_with_logits(ds,torch.ones_like(ds))+
                                   nn.functional.binary_cross_entropy_with_logits(dt,torch.zeros_like(dt)))
            else: loss=nn.functional.cross_entropy(model(x),y,weight=wt)
            loss.backward(); optimizer.step()
    return model,{'optimizer_steps':steps*epochs,'parameter_count':sum(p.numel() for p in model.parameters()),
                  'epochs':epochs,'batch_size':batch,'class_weights':';'.join(map(str,weights))}

def network_output(model,x,classes):
    model.eval()
    with torch.no_grad():
        f=model.fe(torch.from_numpy(np.asarray(x,np.float32)))
        probs=torch.softmax(model.cls(f),1).numpy()
    return np.asarray(classes)[probs.argmax(1)],probs,f.numpy()

def distance_model(f,y):
    params=[]
    for c in np.unique(y):
        z=f[y==c].astype(float); mu=z.mean(0)
        cov=np.cov(z,rowvar=False)+np.eye(z.shape[1])*1e-6
        params.append((mu,np.linalg.pinv(cov,hermitian=True)))
    return params

def distances(f,params):
    out=[]
    for mu,inv in params:
        diff=f.astype(float)-mu
        out.append(np.maximum(0,np.einsum('ij,jk,ik->i',diff,inv,diff)))
    return np.min(out,axis=0)

def seed_summary(metrics,keys):
    """Equal fold mean per seed, then sample SD across seeds; no pseudo-replication."""
    g=metrics.groupby(keys+['seed','metric'],dropna=False)['value']
    per=g.agg(value='mean',n_folds='count').reset_index()
    agg=per.groupby(keys+['metric'],dropna=False)['value'].agg(mean='mean',std=lambda x:x.std(ddof=1),n_seeds='count').reset_index()
    return per,agg
