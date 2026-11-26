import zipfile, io, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
out=Path(__file__).parent
with zipfile.ZipFile('/mnt/data/secom.zip') as z:
    X=pd.read_csv(io.BytesIO(z.read('secom.data')),sep=r'\s+',header=None,na_values=['NaN'])
    Y=pd.read_csv(io.BytesIO(z.read('secom_labels.data')),sep=r'\s+',header=None)
y=(Y.iloc[:,0]==1).astype(int)
assert X.shape[0]==len(y)
missing=X.isna().mean(); const=(X.nunique(dropna=True)<=1)
tr,temp=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=20261122)
va,te=train_test_split(temp,test_size=.5,stratify=y.iloc[temp],random_state=20261122)
# Drop all-NaN and constant columns based on training only
keep=X.iloc[tr].columns[X.iloc[tr].notna().any() & (X.iloc[tr].nunique(dropna=True)>1)]
X=X[keep]
models={
 'prevalence':DummyClassifier(strategy='prior'),
 'logistic':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=3000,random_state=42))]),
 'random_forest':Pipeline([('impute',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=250,min_samples_leaf=3,n_jobs=-1,random_state=42))])
}
rows=[]; preds=[]
for name,m in models.items():
 m.fit(X.iloc[tr],y.iloc[tr]);
 for split,ids in [('validation',va),('test',te)]:
  p=m.predict_proba(X.iloc[ids])[:,1]; yy=y.iloc[ids].to_numpy()
  # Threshold selected by validation only for models; fixed .5 for unbiased diagnostic
  metrics={'model':name,'split':split,'n':len(ids),'positives':int(yy.sum()),'roc_auc':float(roc_auc_score(yy,p)),'pr_auc':float(average_precision_score(yy,p)),'brier':float(brier_score_loss(yy,p)),'log_loss':float(log_loss(yy,np.clip(p,1e-10,1-1e-10))),'precision_at_05':float(precision_score(yy,p>=.5,zero_division=0)),'recall_at_05':float(recall_score(yy,p>=.5,zero_division=0))}
  rows.append(metrics)
  preds.extend({'row_id':int(i),'split':split,'model':name,'actual_fail':int(v),'predicted_fail_probability':float(q)} for i,v,q in zip(ids,yy,p))
pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
pd.DataFrame(preds).to_csv(out/'predictions.csv',index=False)
meta={'source':'User-provided original UCI SECOM ZIP','original_shape':list((len(y),X.shape[1])),'failures':int(y.sum()),'passes':int((1-y).sum()),'overall_missing_fraction':float(missing.mean()),'columns_all_missing':int((X.shape[1]+0) if False else (missing==1).sum()),'columns_constant_or_empty_full_data':int(const.sum()),'columns_retained_train_only':len(keep),'splits':{'train':len(tr),'validation':len(va),'test':len(te)},'seed':20261122,'split_type':'stratified random; potential temporal/batch leakage not ruled out','label_mapping':{'1':'fail','-1':'pass'}}
(out/'dataset_audit.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2));print(pd.DataFrame(rows).to_string(index=False))
