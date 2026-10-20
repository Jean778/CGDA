import unittest
import numpy as np
import pandas as pd
from revision_core import check_disjoint,oscr,ece,seed_summary,quantile_threshold,classification_metrics,stratified_parts

class ProtocolTests(unittest.TestCase):
    def test_overlap_rejected(self):
        with self.assertRaises(AssertionError): check_disjoint(train=[1,2],test=[2,3])
    def test_oscr_perfect(self):
        area,c=oscr(np.array([1,2,9,9]),np.array([1,2,1,1]),np.array([0,1,2,3]),[True,True,False,False])
        self.assertAlmostEqual(area,1)
    def test_oscr_classification_required(self):
        area,c=oscr(np.array([1,2,9,9]),np.array([2,1,1,1]),np.array([0,1,2,3]),[True,True,False,False])
        self.assertAlmostEqual(area,0)
    def test_oscr_ties(self):
        area,c=oscr(np.array([1,2,9,9]),np.array([1,2,1,1]),np.ones(4),[True,True,False,False])
        self.assertAlmostEqual(area,.5)
    def test_f1_evaluation_only(self):
        y=np.array([1,2,1,2]); p=np.array([1,2,0,0]); held=np.array([0,1])
        self.assertEqual(classification_metrics(y[held],p[held])['macro_f1'],1)
    def test_ece_confidence_one(self):
        self.assertEqual(ece(np.array([[1.,0.]]),np.array([2]),[1,2]),1)
    def test_sample_sd_after_fold_average(self):
        x=pd.DataFrame([{'seed':s,'method':'m','metric':'acc','value':v} for s,v in [(1,0),(1,1),(2,1),(2,1)]])
        per,agg=seed_summary(x,['method'])
        self.assertAlmostEqual(agg['mean'].iloc[0],.75)
        self.assertAlmostEqual(agg['std'].iloc[0],np.std([.5,1],ddof=1))
    def test_threshold_not_test_dependent(self):
        self.assertEqual(quantile_threshold(np.array([0.,1.,2.]),.9),2)
    def test_splits_cover_and_disjoint(self):
        y=np.repeat([1,2],10); parts=stratified_parts(np.arange(20),y,[.2,.2,.6],42)
        check_disjoint(a=parts[0],b=parts[1],c=parts[2])
        self.assertEqual(len(np.unique(np.concatenate(parts))),20)

if __name__=='__main__': unittest.main()
