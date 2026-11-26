import io,zipfile,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_score,recall_score,confusion_matrix
from sklearn.isotonic import IsotonicRegression
from scipy.stats import beta
O=Path(__file__).resolve().parent; SEED=20261122
with zipfile.ZipFile(str(O/'secom.zip') if (O/'secom.zip').exists() else '/mnt/data/secom.zip') as z:
 X=pd.read_csv(io.BytesIO(z.read('secom.data')),sep=r'\s+',header=None,na_values=['NaN']); labels=pd.read_csv(io.BytesIO(z.read('secom_labels.data')),sep=r'\s+',header=None)
y=(labels[0]==1).astype(int).to_numpy(); ts=pd.to_datetime(labels[1],format='%d/%m/%Y %H:%M:%S'); order=np.argsort(ts.to_numpy(),kind='stable'); n=len(y)
# Chronological split, no shuffle. Separate from prior F5.1 random split.
tr=order[:int(.6*n)];va=order[int(.6*n):int(.8*n)];te=order[int(.8*n):]
keep=X.iloc[tr].columns[(X.iloc[tr].notna().any())&(X.iloc[tr].nunique(dropna=True)>1)]
X=X[keep]
models={'prevalence':DummyClassifier(strategy='prior'),'logistic':Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=2000,random_state=SEED))]),'random_forest':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=200,min_samples_leaf=3,n_jobs=-1,random_state=SEED))])}
rows=[];pred=[];bootstrap=[]; rng=np.random.default_rng(SEED)
for name,model in models.items():
 model.fit(X.iloc[tr],y[tr]); pv=model.predict_proba(X.iloc[va])[:,1]; pt=model.predict_proba(X.iloc[te])[:,1]
 # select recall-oriented threshold on validation maximizing F2, ties use higher threshold
 thresholds=np.unique(np.r_[0,np.linspace(0.01,0.99,99),pv,1]); scores=[]
 for threshold in thresholds:
  yh=pv>=threshold; tp=((yh==1)&(y[va]==1)).sum(); fp=((yh==1)&(y[va]==0)).sum(); fn=((yh==0)&(y[va]==1)).sum(); score=5*tp/(5*tp+4*fn+fp) if (5*tp+4*fn+fp) else 0
  scores.append(score)
 threshold=float(thresholds[np.argmax(scores)])
 # calibration on validation only; test untouched; isotonic if enough positives
 cal=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(pv,y[va]) if name!='prevalence' else None
 for variant,p in [('raw',pt),('isotonic',cal.predict(pt) if cal else pt)]:
  for thname,th in [('validation_f2',threshold),('default_0.5',.5)]:
   yh=p>=th; tn,fp,fn,tp=confusion_matrix(y[te],yh,labels=[0,1]).ravel()
   rows.append(dict(model=name,calibration=variant,threshold_policy=thname,threshold=th,test_n=len(te),test_fail=int(y[te].sum()),roc_auc=roc_auc_score(y[te],p),pr_auc=average_precision_score(y[te],p),brier=brier_score_loss(y[te],p),tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),precision=precision_score(y[te],yh,zero_division=0),recall=recall_score(y[te],yh,zero_division=0)))
  pred.extend(dict(row_id=int(idx),timestamp=str(ts.iloc[idx]),actual=int(y[idx]),model=name,calibration=variant,p=float(pr)) for idx,pr in zip(te,p))
  # percentile bootstrap on fixed predictions (no model refitting); report 95% interval
  for metric in ['roc_auc','pr_auc','brier']:
   vals=[]
   for _ in range(500):
    ix=rng.integers(0,len(te),len(te)); yy=y[te][ix]; pp=p[ix]
    if metric!='brier' and len(np.unique(yy))<2:continue
    vals.append({'roc_auc':roc_auc_score,'pr_auc':average_precision_score,'brier':brier_score_loss}[metric](yy,pp))
   bootstrap.append(dict(model=name,calibration=variant,metric=metric,estimate={'roc_auc':roc_auc_score,'pr_auc':average_precision_score,'brier':brier_score_loss}[metric](y[te],p),ci_low=np.quantile(vals,.025),ci_high=np.quantile(vals,.975),bootstrap_n=len(vals)))
pd.DataFrame(rows).to_csv(O/'f52_metrics.csv',index=False);pd.DataFrame(pred).to_csv(O/'f52_predictions.csv',index=False);pd.DataFrame(bootstrap).to_csv(O/'f52_bootstrap_ci.csv',index=False)
meta={'dataset_sha256':hashlib.sha256(Path('/mnt/data/secom.zip').read_bytes()).hexdigest(),'dataset':'user-supplied UCI SECOM','original_rows':n,'original_features':590,'retained_features_train_only':len(keep),'split':'chronological 60/20/20 sorted by parsed timestamps','split_n':{'train':len(tr),'validation':len(va),'test':len(te)},'split_fail':{'train':int(y[tr].sum()),'validation':int(y[va].sum()),'test':int(y[te].sum())},'time_ranges':{'train':[str(ts.iloc[tr].min()),str(ts.iloc[tr].max())],'validation':[str(ts.iloc[va].min()),str(ts.iloc[va].max())],'test':[str(ts.iloc[te].min()),str(ts.iloc[te].max())]},'calibration':'isotonic fitted on validation predictions only; validation also used for threshold selection, possible calibration/threshold overfitting','bootstrap':'500 resamples of fixed test predictions; no model refit; temporal dependence ignored','limitations':['Observational quality prediction only; no intervention or rollback records','No group/batch IDs; timestamp ordering alone cannot exclude batch leakage','Validation used for both calibration and threshold tuning; future work should separate','No causal effects inferred']}
(O/'f52_audit.json').write_text(json.dumps(meta,indent=2));print('F52 AUDIT',json.dumps(meta['split_fail']));print(pd.DataFrame(rows).query("calibration=='raw' and threshold_policy=='validation_f2'")[['model','roc_auc','pr_auc','brier','threshold','tp','fp','fn']].to_string(index=False))
