import csv,json
from scipy.stats import beta
from pathlib import Path
out=Path(__file__).parent
rows=[]
for delta in [0,.01,.02,.04,.08,.12]:
 for failures in [0,1,2]:
  threshold=.12-delta
  n=next((n for n in range(failures,10001) if threshold>0 and beta.sf(threshold,1+failures,9+n-failures)<=.10),None)
  rows.append(dict(delta=delta,observed_failures=failures,minimum_total_pilot_observations=n if n is not None else 'infeasible_within_10000',rule='posterior Pr(p > 0.12-delta) <= 0.10'))
with (out/'certificate_thresholds.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
checks=[]
for delta in [0,.01,.02,.04,.08]:
 n=next(int(r['minimum_total_pilot_observations']) for r in rows if r['delta']==delta and r['observed_failures']==0)
 tail=beta.sf(.12-delta,1,9+n)
 prior=beta.sf(.12-delta,1,9+n-1)
 checks.append(dict(delta=delta,n=n,tail=tail,previous_tail=prior,pass_check=tail<=.1 and prior>.1))
with (out/'boundary_checks.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=checks[0]);w.writeheader();w.writerows(checks)
(out/'verification.json').write_text(json.dumps({'checks_passed':all(r['pass_check'] for r in checks),'check_count':len(checks),'scope':'Numerical threshold checks, not algorithmic novelty or empirical safety certification'},indent=2))
print('checks:',len(checks),'passed:',all(bool(r['pass_check']) for r in checks))
