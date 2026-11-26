"""Raw UCI SECOM ZIP -> stratified and chronological/expanding-window metrics."""
import argparse,io,zipfile,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

def run(archive,out):
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        X=pd.read_csv(io.BytesIO(z.read('secom.data')),sep=r'\s+',header=None,na_values=['NaN'])
        labels=pd.read_csv(io.BytesIO(z.read('secom_labels.data')),sep=r'\s+',header=None)
    y=(labels.iloc[:,0]==1).astype(int).to_numpy()
    ts=pd.to_datetime(labels.iloc[:,1],format='%d/%m/%Y %H:%M:%S')
    order=np.argsort(ts.to_numpy(),kind='stable');n=len(y)
    assert X.shape==(1567,590) and y.sum()==104
    tr,tmp=train_test_split(np.arange(n),test_size=.4,stratify=y,random_state=20261122)
    va,te=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=20261122)
    folds=[('random_test',tr,te),('chronological_test',order[:int(.6*n)],order[int(.8*n):])]
    folds += [(f'expanding_{i}',order[:a],order[a:b]) for i,(a,b) in enumerate([(900,1060),(1060,1220),(1220,1380),(1380,1567)],1)]
    metrics=[];preds=[]
    for fold,train,test in folds:
        keep=X.iloc[train].columns[X.iloc[train].notna().any() & (X.iloc[train].nunique(dropna=True)>1)]
        models={'prevalence':DummyClassifier(strategy='prior'),
            'logistic':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('clf',LogisticRegression(C=.1,max_iter=2000,random_state=42))]),
            'random_forest':Pipeline([('impute',SimpleImputer(strategy='median')),('clf',RandomForestClassifier(n_estimators=150 if fold.startswith('expanding') else 200,min_samples_leaf=3,n_jobs=-1,random_state=42))])}
        for name,model in models.items():
            model.fit(X.iloc[train][keep],y[train]);p=model.predict_proba(X.iloc[test][keep])[:,1];yy=y[test]
            metrics.append(dict(fold=fold,model=name,train_n=len(train),test_n=len(test),test_failures=int(yy.sum()),roc_auc=roc_auc_score(yy,p),pr_auc=average_precision_score(yy,p),brier=brier_score_loss(yy,p)))
            preds.extend(dict(fold=fold,model=name,row_id=int(i),actual=int(v),predicted=float(q)) for i,v,q in zip(test,yy,p))
    pd.DataFrame(metrics).to_csv(out/'secom_integrated_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'secom_integrated_predictions.csv',index=False)
    (out/'secom_integrated_audit.json').write_text(json.dumps({'dataset_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'rows':n,'features':590,'failures':int(y.sum()),'folds':[x[0] for x in folds],'note':'Models refit on each training window; this is not the identical hyperparameter protocol used in every earlier phase.'},indent=2))
    return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--secom-zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(pd.DataFrame(run(a.secom_zip,a.output)).to_string(index=False))
