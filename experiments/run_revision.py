"""CGDA revision suite. CSV-only numerical outputs; resumable, hashed and isolated."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[_k]='1'
import argparse,hashlib,json,math,sys,time,traceback,platform,shutil
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
import sklearn
from revision_core import *

VERSION='cgda_protocol_v2_20260909'
MAIN=['base','cal_sup','cg_mix','cg_sup','mixup_std','cg_shuffle','dann','logreg','sklearn_mlp','mean_shift']
SINGLE_SOURCE=['base','cal_sup','cg_sup','cg_shuffle','dann']
KEYS=['family','setting','budget','method','threshold_rule','evaluation']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv(df,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(df); temp=path.with_suffix('.csv.tmp')
    df.to_csv(temp,index=False,float_format='%.12g'); temp.replace(path)

@lru_cache(None)
def load_drift(project):
    d=pd.read_csv(Path(project)/'data/processed/drift_clean.csv')
    return np.log1p(np.maximum(d.filter(regex='^s').values,0)).astype(np.float32),d.gas_id.values,d.batch.values

@lru_cache(None)
def load_gsa(project):
    d=pd.read_csv(Path(project)/'data/raw/gsalc.csv',header=None)
    names=['ethanol','acetone','toluene','ethyl acetate','isopropanol','hexane']
    y=np.array([names.index(s) for s in d.iloc[:,0]]); conc=d.iloc[:,1].str.replace('ppb','').astype(int).values
    ts=d.iloc[:,2:].values.reshape(len(d),10,900); feats=[]
    for s in range(10):
        z=ts[:,s,:]; b=z[:,:45].mean(1); t=np.arange(900); centered=t-t.mean()
        slope=z@centered/(centered@centered)
        feats.extend([z.mean(1),z.std(1),z.min(1),z.max(1),b,z.max(1)-b,slope])
    return np.stack(feats,1),y,conc

@lru_cache(None)
def load_aq(project):
    raw=pd.read_csv(Path(project)/'data/raw/AirQualityUCI.csv',sep=';',decimal=',').iloc[:,:15]
    sensors=['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)']
    refs=['CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)']
    d=raw.replace(-200.,np.nan).dropna(subset=sensors+refs+['T','RH']).copy()
    d['timestamp']=pd.to_datetime(d['Date']+' '+d['Time'],format='%d/%m/%Y %H.%M.%S')
    d=d.sort_values('timestamp'); ids=d.index.values
    b=(d['T'].values>=20).astype(int)*2+(d['RH'].values>=50).astype(int)
    return d[sensors].values,d[refs].values,b,ids,d.timestamp.astype(str).values

class Recorder:
    def __init__(self,task,out): self.t=task; self.out=out; self.metrics=[]; self.config=[]
    def metric(self,method,values,threshold_rule='none',evaluation='heldout',**extras):
        for k,v in values.items():
            self.metrics.append({**self.t,'method':method,'threshold_rule':threshold_rule,'evaluation':evaluation,
                                 'metric':k,'value':float(v),**extras})
    def configrow(self,method,**kwargs): self.config.append({**self.t,'method':method,**kwargs})
    def finish(self): csv(self.metrics,self.out/'metrics.csv'); csv(self.config,self.out/'model_details.csv')
    def predictions(self,method,ids,y,p,pr=None,**cols):
        d={'sample_id':ids,'y_true':y,'y_pred':p,**cols}
        if pr is not None:
            for i in range(pr.shape[1]): d[f'p_class_position_{i}']=pr[:,i]
        csv(d,self.out/f'predictions_{method}.csv')
    def split(self,**parts):
        rows=[{'role':k,'sample_id':int(i)} for k,z in parts.items() for i in z]
        csv(rows,self.out/'splits.csv')

def shuffled_pool(pool,seed):
    z=pool.copy(); np.random.RandomState(seed+888).shuffle(z); return z

def neural_or_lr(xs,ys,xc,yc,xt,method,seed,classes,epochs,steps):
    if method in ('logreg','sklearn_mlp'):
        obj=LogisticRegression(C=1.,max_iter=2000) if method=='logreg' else MLPClassifier(hidden_layer_sizes=(128,128,64),max_iter=500,random_state=seed)
        obj.fit(xs,ys); raw=obj.predict_proba(xt); pr=np.zeros((len(xt),len(classes)))
        for j,c in enumerate(obj.classes_): pr[:,list(classes).index(c)]=raw[:,j]
        return np.asarray(classes)[pr.argmax(1)],pr,{'iterations':int(np.max(obj.n_iter_))},None
    model,details=train_network(xs,ys,xc,yc,method,seed,classes,epochs,steps)
    p,pr,f=network_output(model,xt,classes)
    return p,pr,details,model

def run_closed(task,project,out,epochs):
    x,y,b=load_drift(project); target=np.flatnonzero(b==task['fold']); seed=task['seed']
    if task['family']=='forward': source=np.flatnonzero(b<task['fold'])
    elif task['family']=='singlesource': source=np.flatnonzero(b==int(task['setting'][1:]))
    else: source=np.flatnonzero(b!=task['fold'])
    pool,test=stratified_parts(target,y,[.2,.8],seed)
    pool=shuffled_pool(pool,seed); cal=nested_calibration(pool,target,y,task['budget'])
    check_disjoint(source=source,calibration=cal,test=test)
    sc=StandardScaler().fit(x[source]); xs=sc.transform(x[source]).astype(np.float32)
    xc=sc.transform(x[cal]).astype(np.float32); xt=sc.transform(x[test]).astype(np.float32)
    steps=math.ceil((len(source)+len(pool)+360)/128)
    methods=MAIN if task['family']=='closed' else (SINGLE_SOURCE if task['family']=='singlesource' else ['base','cal_sup','cg_sup'])
    r=Recorder(task,out); r.split(source_train=source,calibration_train=cal,test=test,unused_pool=np.setdiff1d(pool,cal))
    csv({'feature':np.arange(x.shape[1]),'mean':sc.mean_,'scale':sc.scale_},out/'preprocessing.csv')
    base=None
    for m in methods:
        t=time.time()
        if m=='mean_shift':
            assert base is not None
            shift=xc.mean(0)-xs.mean(0); p,pr,_=network_output(base,xt-shift,np.arange(1,7)); details={'reused_model':'base'}
        else:
            p,pr,details,model=neural_or_lr(xs,y[source],xc,y[cal],xt,m,seed,np.arange(1,7),epochs,steps)
            if m=='base': base=model
        r.metric(m,classification_metrics(y[test],p,pr,np.arange(1,7)),n_test=len(test))
        r.predictions(m,test,y[test],p,pr)
        r.configrow(m,n_source=len(source),n_target_labeled=len(cal) if m in ['cal_sup','cg_mix','cg_sup','cg_shuffle'] else 0,
                    n_target_unlabeled=len(cal) if m in ['dann','mean_shift'] else 0,n_test=len(test),
                    seconds=time.time()-t,**details)
    r.finish()

def run_osr(task,project,out,epochs):
    x,y,b=load_drift(project); seed=task['seed']; target=np.flatnonzero(b==task['fold'])
    unknown=np.array([int(c) for c in task['setting'].split('-')]); classes=np.array([c for c in range(1,7) if c not in unknown])
    source=np.flatnonzero((b!=task['fold'])&np.isin(y,classes))
    train,sval=stratified_parts(source,y,[.9,.1],seed+33)
    pool,thpool,test=stratified_parts(target,y,[.2,.2,.6],seed)
    cal=nested_calibration(shuffled_pool(pool,seed),target,y,.1); cal=cal[np.isin(y[cal],classes)]
    check_disjoint(source_train=train,source_threshold=sval,calibration_train=cal,target_threshold=thpool,test=test)
    sc=StandardScaler().fit(x[train]); xs=sc.transform(x[train]).astype(np.float32)
    xc=sc.transform(x[cal]).astype(np.float32); xt=sc.transform(x[test]).astype(np.float32)
    xv=sc.transform(x[sval]).astype(np.float32); xp=sc.transform(x[thpool]).astype(np.float32)
    k=np.isin(y[test],classes); kp=np.isin(y[thpool],classes)
    if not k.any() or not (~k).any(): raise ValueError('No known or unknown test examples')
    r=Recorder(task,out); r.split(source_train=train,source_threshold=sval,calibration_train=cal,target_threshold=thpool,test=test,unused_pool=np.setdiff1d(pool,cal))
    csv({'feature':np.arange(x.shape[1]),'mean':sc.mean_,'scale':sc.scale_},out/'preprocessing.csv')
    thresholds=[]; sampling=[]
    for mode in ['base','cal_sup','cg_sup']:
        model,details=train_network(xs,y[train],xc,y[cal],mode,seed,classes,epochs,math.ceil((len(train)+len(pool)+240)/128))
        p,pr,ft=network_output(model,xt,classes); _,_,fs=network_output(model,xs,classes)
        _,_,fv=network_output(model,xv,classes); pp,ppr,fp=network_output(model,xp,classes)
        params=distance_model(fs,y[train]); dt=distances(ft,params); dv=distances(fv,params); dp=distances(fp,params)
        area,curve=oscr(y[test],p,dt,k); csv(curve,out/f'oscr_curve_{mode}.csv')
        r.metric(mode,{'auroc':roc_auc_score(k.astype(int),-dt),'oscr':area,
                       'known_accuracy':accuracy_score(y[test][k],p[k]),'known_macro_f1':f1_score(y[test][k],p[k],labels=np.unique(y[test][k]),average='macro',zero_division=0)})
        r.predictions(mode,test,y[test],p,pr,distance=dt,is_known=k.astype(int))
        csv({'sample_id':thpool,'distance':dp,'is_known':kp.astype(int),'y_true':y[thpool]},out/f'threshold_scores_{mode}.csv')
        csv({'sample_id':sval,'distance':dv,'y_true':y[sval]},out/f'source_validation_scores_{mode}.csv')
        rules={'source_q90':(quantile_threshold(dv,.9),len(dv),0.)}
        if kp.any(): rules['known_reference_q90']=(quantile_threshold(dp[kp],.9),int(kp.sum()),0.)
        # Fixed size across contamination ratios within this fold; labels used only
        # by benchmark sampling, never by mixed-pool threshold estimation.
        n=(min(100,int(kp.sum()),int((~kp).sum()/.75))//4)*4
        if n>=4:
            rng=np.random.RandomState(seed+task['fold']*17); ki=rng.permutation(np.flatnonzero(kp)); ui=rng.permutation(np.flatnonzero(~kp))
            for rho in [0.,.25,.5,.75]:
                nu=int(n*rho); sel=np.r_[ki[:n-nu],ui[:nu]]; mix=dp[sel]
                tail=mix[mix<=np.median(mix)]
                tag=f'{int(rho*100):02d}'
                rules[f'mixed{tag}_full_q90']=(quantile_threshold(mix,.9),n,rho)
                rules[f'mixed{tag}_tail_q90']=(quantile_threshold(tail,.9),n,rho)
                if mode=='base': sampling.extend({'contamination':rho,'sample_id':int(thpool[i]),'is_known':int(kp[i])} for i in sel)
        for name,(thr,nref,rho) in rules.items():
            r.metric(mode,rejection_metrics(y[test],p,dt,k,thr),threshold_rule=name,n_test=len(test))
            thresholds.append({'method':mode,'threshold_rule':name,'threshold':thr,'n_reference':nref,'unknown_fraction':rho})
        r.configrow(mode,n_source=len(train),n_calibration_train=len(cal),n_threshold_pool=len(thpool),n_mixed=n,
                    known_classes=';'.join(map(str,classes)),unknown_classes=task['setting'],**details)
    csv(thresholds,out/'thresholds.csv'); csv(sampling,out/'mixed_pool_membership.csv'); r.finish()

def run_gsa(task,project,out,epochs):
    x,y,con=load_gsa(project); seed=task['seed']; rng=np.random.RandomState(seed)
    if task['family']=='gsa':
        target=np.flatnonzero(con==task['fold']); train=np.flatnonzero(con!=task['fold'])
    else:
        from sklearn.model_selection import StratifiedKFold
        train,target=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y))[task['fold']]
    cal=np.concatenate([rng.choice(target[y[target]==c],size=min(2,len(target[y[target]==c])-1),replace=False) for c in np.unique(y[target])])
    test=np.setdiff1d(target,cal); check_disjoint(source_train=train,calibration=cal,test=test)
    sc=StandardScaler().fit(x[train]); xs=sc.transform(x[train]); xc=sc.transform(x[cal]); xt=sc.transform(x[test])
    r=Recorder(task,out); r.split(source_train=train,calibration=cal,test=test)
    csv({'feature':np.arange(x.shape[1]),'mean':sc.mean_,'scale':sc.scale_},out/'preprocessing.csv')
    for mode in ['base','cal_sup','cg_sup']:
        xb=xs.copy(); yb=y[train].copy()
        if mode!='base': xb=np.r_[xb,xc]; yb=np.r_[yb,y[cal]]
        if mode=='cg_sup':
            gen=np.random.RandomState(seed); xx=[]; yy=[]
            for c in np.unique(y[cal]):
                ids=np.flatnonzero(y[train]==c); center=.5*xs[ids].mean(0)+.5*xc[y[cal]==c].mean(0)
                for _ in range(10):
                    i=gen.choice(ids); lam=gen.beta(.5,.5)*.6
                    xx.append((1-lam)*xs[i]+lam*center); yy.append(c)
            xb=np.r_[xb,np.stack(xx)]; yb=np.r_[yb,yy]
        model=LogisticRegression(C=1.,max_iter=2000).fit(xb,yb); pr=model.predict_proba(xt); p=model.predict(xt)
        r.metric(mode,classification_metrics(y[test],p,pr,model.classes_),n_test=len(test))
        r.predictions(mode,test,y[test],p,pr)
        r.configrow(mode,n_source=len(train),n_calibration=len(cal) if mode!='base' else 0,n_test=len(test),classifier='LogisticRegression',iterations=model.n_iter_[0])
    r.finish()

def chronological_domain_blocks(b):
    return {i:tuple(np.split(np.flatnonzero(b==i),[int(.6*np.sum(b==i)),int(.8*np.sum(b==i))])) for i in range(4)}

def aq_setup(project,A,B,seed):
    x,c,b,rawids,times=load_aq(project); blocks=chronological_domain_blocks(b)
    atr,ava,ate=blocks[A]; btr,bva,bte=blocks[B]
    low=np.quantile(c[atr],.5,axis=0); high=np.quantile(c[atr],.95,axis=0)
    normal=(c<=low).all(1); event=(c>high).any(1); keep=normal|event
    rng=np.random.RandomState(seed)
    cn=btr[normal[btr]]; ct=bva[normal[bva]]
    if min(len(cn),len(ct))<3: raise ValueError(f'insufficient normal reference: {len(cn)}, {len(ct)}')
    cn=rng.choice(cn,min(15,len(cn)),replace=False); ct=rng.choice(ct,min(15,len(ct)),replace=False)
    train=np.setdiff1d(atr[keep[atr]],np.r_[cn,ct]); val=np.setdiff1d(ava[normal[ava]],np.r_[cn,ct])
    test=bte[keep[bte]]; broad=bte
    if len(np.unique(event[test]))<2 or len(val)<3 or len(np.unique(event[train]))<2: raise ValueError('insufficient source or test label support')
    check_disjoint(source_train=train,source_threshold=val,target_train_calibration=cn,target_threshold_calibration=ct,test=broad)
    return x,c,b,rawids,times,train,val,cn,ct,test,broad,normal,event,low,high

def run_aq(task,project,out,epochs):
    A,B=map(int,task['setting'].split('-')); seed=task['seed']
    x,c,b,rawids,times,tr,sv,cn,ct,te,broad,normal,event,low,high=aq_setup(project,A,B,seed)
    sc=StandardScaler().fit(x[tr]); xs=sc.transform(x[tr]).astype(np.float32)
    xv=sc.transform(x[sv]).astype(np.float32); xc=sc.transform(x[cn]).astype(np.float32); xct=sc.transform(x[ct]).astype(np.float32)
    xb=sc.transform(x[broad]).astype(np.float32); yt=event[tr].astype(int); in_strict=np.isin(broad,te)
    r=Recorder(task,out); r.split(source_train=rawids[tr],source_threshold=rawids[sv],target_train_calibration=rawids[cn],target_threshold_calibration=rawids[ct],test=rawids[broad])
    csv({'feature':np.arange(x.shape[1]),'mean':sc.mean_,'scale':sc.scale_},out/'preprocessing.csv')
    csv({'reference':['CO','C6H6','NOx','NO2'],'normal_p50':low,'event_p95':high},out/'label_thresholds.csv')
    scores={}; thresholds=[]
    pca=PCA(n_components=.95).fit(xs[yt==0])
    recon=lambda z:np.square(z-pca.inverse_transform(pca.transform(z))).sum(1)
    scores['pca']=(recon(xb),quantile_threshold(recon(xv),.95))
    steps=math.ceil((len(tr)+30+40)/128)
    for mode in ['base','cal_sup','cg_sup']:
        model,details=train_network(xs,yt,xc,np.zeros(len(cn),int),mode,seed,[0,1],epochs=80 if epochs==50 else epochs,steps=steps,small=True)
        _,_,fs=network_output(model,xs,[0,1]); _,_,fv=network_output(model,xv,[0,1]); _,_,fc=network_output(model,xc,[0,1]); _,_,fct=network_output(model,xct,[0,1]); _,_,fb=network_output(model,xb,[0,1])
        params=distance_model(fs[yt==0],np.zeros(np.sum(yt==0),int)); shift=fc.mean(0)-params[0][0]
        ds=distances(fb,params); dct=distances(fct,params); dsv=distances(fv,params)
        shifted=distances(fb-shift,params); shiftedct=distances(fct-shift,params)
        src_thr=quantile_threshold(dsv,.95); tgt_thr=quantile_threshold(dct,.95); shift_thr=quantile_threshold(shiftedct,.95)
        if mode=='base':
            scores.update({'mlp':(ds,src_thr),'threshold_only':(ds,tgt_thr),'shift_only':(shifted,src_thr),'shift_threshold':(shifted,shift_thr)})
        elif mode=='cal_sup': scores.update({'cal_sup':(ds,src_thr),'cal_sup_threshold':(ds,tgt_thr)})
        else: scores.update({'cg_train':(ds,src_thr),'cg_train_threshold':(ds,tgt_thr),'cg_full':(shifted,shift_thr)})
        csv({'sample_id':rawids[ct],'distance_unshifted':dct,'distance_shifted':shiftedct},out/f'threshold_scores_{mode}.csv')
        csv({'sample_id':rawids[sv],'distance':dsv},out/f'source_validation_scores_{mode}.csv')
        csv({'feature':np.arange(len(shift)),'shift':shift},out/f'feature_shift_{mode}.csv')
        r.configrow(mode,n_source=len(tr),n_source_normal=int(np.sum(yt==0)),n_calibration_training=len(cn),n_calibration_threshold=len(ct),n_strict_test=len(te),n_all_test=len(broad),source_domain=A,target_domain=B,**details)
    for method,(d,thr) in scores.items():
        for name,mask in [('extreme',in_strict),('including_intermediate',np.ones(len(broad),bool))]:
            y=event[broad][mask].astype(int)
            r.metric(method,binary_metrics(y,d[mask],thr),evaluation=name,n_test=int(mask.sum()))
        r.predictions(method,rawids[broad],event[broad].astype(int),(d>thr).astype(int),distance=d,in_extreme_task=in_strict.astype(int),timestamp=times[broad],threshold=thr)
        thresholds.append({'method':method,'threshold':thr})
    csv(thresholds,out/'thresholds.csv'); r.finish()

def task_list():
    tasks=[]
    def add(family,setting,budget,folds):
        for seed in SEEDS:
            for fold in folds: tasks.append(dict(family=family,setting=setting,budget=budget,seed=seed,fold=fold))
    add('closed','lobo',.1,range(1,11))
    for frac in [.05,.2]: add('budget','lobo',frac,range(1,11))
    add('forward','past_to_future',.1,range(2,11))
    add('singlesource','b1',.1,range(2,11))
    for u in ['4-6','1-2','3-5']: add('osr',u,.1,range(1,11))
    add('gsa','concentration',2,[50,100,200]); add('gsa_cv','stratified_cv',2,range(5))
    for A in range(4):
        for B in range(4): add('aq',f'{A}-{B}',30,[0])
    return tasks

def task_id(t): return f"{t['family']}_{t['setting']}_b{t['budget']}_s{t['seed']}_f{t['fold']}"

def code_hash():
    return hashlib.sha256((''.join(sha(Path(__file__).parent/x) for x in ['revision_core.py','run_revision.py'])).encode()).hexdigest()

def execute_task(task,project,root,epochs=50):
    out=Path(root)/'tasks'/task_id(task); out.mkdir(parents=True,exist_ok=True)
    done=out/'completion.csv'; digest=code_hash()
    if done.exists():
        old=pd.read_csv(done)
        if old.iloc[0]['code_sha256']!=digest: raise RuntimeError('Code changed after completed task; choose new output root')
        return task_id(task),'cached',float(old.iloc[0].seconds)
    started=time.time(); csv([{**task,'code_sha256':digest,'epochs':epochs}],out/'task_config.csv')
    try:
        if task['family'] in ['closed','budget','forward','singlesource']: run_closed(task,project,out,epochs)
        elif task['family']=='osr': run_osr(task,project,out,epochs)
        elif task['family'] in ['gsa','gsa_cv']: run_gsa(task,project,out,epochs)
        else: run_aq(task,project,out,epochs)
        csv([{'status':'complete','code_sha256':digest,'seconds':time.time()-started}],done)
        return task_id(task),'complete',time.time()-started
    except Exception as e:
        csv([{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'seconds':time.time()-started}],out/'failure.csv')
        if task['family']=='aq' and 'insufficient' in str(e):
            csv([{'status':'ineligible','reason':str(e),'code_sha256':digest,'seconds':time.time()-started}],out/'ineligible.csv')
            return task_id(task),'ineligible',time.time()-started
        return task_id(task),'failed',time.time()-started

def aggregate(root):
    root=Path(root); frames=[]; status=[]
    for t in task_list():
        path=root/'tasks'/task_id(t); complete=(path/'completion.csv').exists()
        status.append({**t,'task_id':task_id(t),'status':'complete' if complete else 'ineligible' if (path/'ineligible.csv').exists() else 'failed' if (path/'failure.csv').exists() else 'pending'})
        if complete: frames.append(pd.read_csv(path/'metrics.csv'))
    csv(status,root/'task_status.csv')
    if not frames:return
    data=pd.concat(frames,ignore_index=True); csv(data,root/'metrics_by_fold_seed.csv')
    per,agg=seed_summary(data,KEYS); csv(per,root/'metrics_by_seed.csv'); csv(agg,root/'summary_mean_sd.csv')
    # Per-fold statistics allow every repeated domain-specific result to carry SD too.
    _,fold=seed_summary(data,KEYS+['fold']); csv(fold,root/'summary_by_fold_mean_sd.csv')
    # Air Quality non-diagonal pairs are equal-weight within each seed; diagonal separately.
    aq=data[data.family=='aq'].copy()
    if len(aq):
        aq['setting']=aq.setting.map(lambda v:'same_domain' if v.split('-')[0]==v.split('-')[1] else 'cross_domain')
        a,b=seed_summary(aq,KEYS); csv(a,root/'aq_domain_metrics_by_seed.csv'); csv(b,root/'aq_domain_summary_mean_sd.csv')
    # Paired differences, matched seed/fold/metric, rather than subtraction of unrelated means.
    diffs=[]
    for family,base,comparison in [('closed','base','cg_sup'),('closed','cal_sup','cg_sup'),('singlesource','base','cg_sup'),('singlesource','cal_sup','cg_sup'),('gsa','cal_sup','cg_sup'),('gsa','base','cal_sup')]:
        sub=data[(data.family==family)&(data.threshold_rule=='none')]
        ix=['family','setting','budget','seed','fold','evaluation','metric']
        wide=sub.pivot_table(index=ix,columns='method',values='value').reset_index()
        if base in wide and comparison in wide:
            wide['value']=wide[comparison]-wide[base];wide['method']=comparison+'-minus-'+base;wide['threshold_rule']='none';diffs.append(wide[ix+['value','method','threshold_rule']])
    if diffs:
        d=pd.concat(diffs); p,a=seed_summary(d,KEYS);csv(d,root/'paired_differences_by_fold.csv');csv(p,root/'paired_differences_by_seed.csv');csv(a,root/'paired_differences_mean_sd.csv')

def freeze(project,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in [Path(project)/'data/processed/drift_clean.csv',Path(project)/'data/raw/gsalc.csv',Path(project)/'data/raw/AirQualityUCI.csv']:
        rows.append({'file':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    csv(rows,root/'data_manifest.csv')
    csv([{'protocol':VERSION,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'code_sha256':code_hash(),'seeds':';'.join(map(str,SEEDS)),'std_ddof':1}],root/'environment.csv')
    csv(task_list(),root/'planned_tasks.csv')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='REPO');ap.add_argument('--out',required=True)
    ap.add_argument('--family',default='all');ap.add_argument('--workers',type=int,default=3);ap.add_argument('--smoke',action='store_true');ap.add_argument('--aggregate',action='store_true');ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args()
    if args.aggregate: aggregate(args.out);return
    if args.preflight:
        rows=[]
        for A in range(4):
            for B in range(4):
                try:
                    v=aq_setup(args.project,A,B,42); rows.append({'source':A,'target':B,'train':len(v[5]),'source_validation':len(v[6]),'cal_train':len(v[7]),'cal_threshold':len(v[8]),'test':len(v[9]),'status':'ok'})
                except Exception as e: rows.append({'source':A,'target':B,'status':str(e)})
        csv(rows,Path(args.out)/'preflight_counts.csv');print(pd.DataFrame(rows).to_string(index=False));return
    freeze(args.project,args.out)
    tasks=task_list()
    if args.family!='all':tasks=[t for t in tasks if t['family'] in args.family.split(',')]
    if args.smoke:
        selected=[]
        for f in ['closed','osr','gsa','aq','singlesource']:
            selected.extend([t for t in tasks if t['family']==f][:1])
        tasks=selected
    start=time.time(); print(f'PLAN tasks={len(tasks)} workers={args.workers} smoke={args.smoke}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute_task,t,args.project,args.out,2 if args.smoke else 50):t for t in tasks}
        for n,f in enumerate(as_completed(futures),1):
            try: ident,status,elapsed=f.result()
            except Exception as e: print(f'WORKER ERROR {repr(e)}',flush=True);raise
            print(f'[{n}/{len(tasks)}] {status} {ident} seconds={elapsed:.1f} wall={time.time()-start:.1f}',flush=True)
    aggregate(args.out)
    print('FINISHED',flush=True)

if __name__=='__main__':main()
