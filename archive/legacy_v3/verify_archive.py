from pathlib import Path
import pandas as pd

r=Path(__file__).resolve().parent
a=pd.read_csv(r/'data_derived/secom_random_metrics.csv')
b=pd.read_csv(r/'data_derived/secom_chronological_metrics.csv')
c=pd.read_csv(r/'data_derived/policy_summary.csv')
d=pd.read_csv(r/'data_derived/f71_f8_raw.csv')
rf=a[(a.model=='random_forest')&(a.split=='test')].iloc[0]
assert abs(rf.roc_auc-0.7536161222168047)<1e-9
rf2=b[(b.model=='random_forest')&(b.calibration=='raw')&(b.threshold_policy=='default_0.5')].iloc[0]
assert 0.38<rf2.roc_auc<0.41
assert len(d)==6000, len(d)
assert len(c)==30, len(c)
print('PASS: core saved outputs and row counts; not an independent scientific audit')
