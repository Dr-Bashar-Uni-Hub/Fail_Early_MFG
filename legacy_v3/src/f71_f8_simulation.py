import csv,random,statistics,math,json,zipfile
from pathlib import Path
P=Path(__file__).parent
# Independent, transparent discrete-event simulator; no use of F4 Bellman oracle.
# Outcomes common across policies by seed, machine, time, batch item.
SC={'nominal':(.035,.025,1,0),'late_shift':(.035,.24,1,4),'high_defect':(.22,.22,1,0),'delayed_reports':(.035,.025,3,0),'rollback_expensive':(.035,.24,1,4)}
POL=['queue_guard','matched_fixed','matched_reactive','no_queue','no_rollback','no_risk_gate']
N=200;H=12;B=20

def sim(seed,sc,pol):
 p0,p1,lag,shift=SC[sc];rng=random.Random(seed)
 U=[[rng.random() for j in range(B)] for t in range(H)]
 t=0;pending=[];obs_bad=0;obs_n=0;mode='pilot';net=0.;pilot_n=0;deploy=False;rollback=False;defects=0;produced=0;unsafe=0;first_deploy=None;report_lags=[]
 # A deployment is permitted only after evidence unless no_risk_gate. Dynamic risk guard uses Wilson upper bound.
 for t in range(H):
  arrivals=[x for x in pending if x[0]<=t];pending=[x for x in pending if x[0]>t]
  for due,k,n,origin in arrivals:obs_bad+=k;obs_n+=n;report_lags.append(due-origin)
  p=p1 if (shift and t>=shift) else p0
  k=sum(u<p for u in U[t]);
  rate=obs_bad/obs_n if obs_n else 1
  upper=(rate+1.96**2/(2*obs_n)+1.96*math.sqrt(rate*(1-rate)/obs_n+1.96**2/(4*obs_n**2)))/(1+1.96**2/obs_n) if obs_n else 1
  # same reports, physical stream and costs for every policy; no future observations.
  if mode=='pilot':
   if pol=='matched_fixed':
    if pilot_n>=2 and obs_n>=40:act='deploy' if rate<=.12 else 'abandon'
    else:act='pilot'
   elif pol=='matched_reactive':
    if obs_n>=40:act='deploy' if upper<=.18 else ('abandon' if pilot_n>=4 else 'pilot')
    else:act='pilot'
   elif pol=='no_risk_gate':act='deploy' if obs_n>=40 else 'pilot'
   else:
    if obs_n>=40 and upper<=.18:act='deploy'
    elif pilot_n>=5 and not pending:act='abandon'
    else:act='pilot'
   if act=='abandon':break
   if act=='deploy':mode='deployed';deploy=True;first_deploy=t;net-=3
   else:
    net-=.5;pilot_n+=1
    # no_queue ablation incorrectly treats report as immediately available
    if pol=='no_queue':obs_bad+=k;obs_n+=B
    else:pending.append((t+lag,k,B,t))
    continue
  if mode=='deployed':
   # React to arrived reports; rollback only on evidence, no access to true p.
   if pol not in ('matched_fixed','no_rollback','no_risk_gate') and obs_n>=40 and rate>.12:
    rollback=True;net-= (18 if sc=='rollback_expensive' else 6);break
   # same realized production rewards and observable reporting queue
   produced+=B;defects+=k;unsafe+=int(p>.12)
   net+=8-35*k/B
   if pol=='no_queue':obs_bad+=k;obs_n+=B
   else:pending.append((t+lag,k,B,t))
 return dict(net=net,pilots=pilot_n,deployed=int(deploy),rollback=int(rollback),defects=defects,produced=produced,unsafe_periods=unsafe,first_deploy=-1 if first_deploy is None else first_deploy,pending_end=len(pending),observed_n=obs_n)
rows=[]
for sc in SC:
 for seed in range(N):
  for pol in POL:rows.append(dict(scenario=sc,seed=seed,policy=pol,**sim(seed,sc,pol)))
def write(name,rs):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
write('f71_f8_raw.csv',rows)
summary=[];paired=[]
for sc in SC:
 for pol in POL:
  rs=[r for r in rows if r['scenario']==sc and r['policy']==pol]
  summary.append(dict(scenario=sc,policy=pol,n=len(rs),mean_net=statistics.mean(r['net'] for r in rs),deploy_rate=statistics.mean(r['deployed'] for r in rs),rollback_rate=statistics.mean(r['rollback'] for r in rs),mean_unsafe_periods=statistics.mean(r['unsafe_periods'] for r in rs),mean_pilots=statistics.mean(r['pilots'] for r in rs)))
 for pol in POL[1:]:
  d=[next(r['net'] for r in rows if r['scenario']==sc and r['seed']==i and r['policy']=='queue_guard')-next(r['net'] for r in rows if r['scenario']==sc and r['seed']==i and r['policy']==pol) for i in range(N)]
  m=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(N)
  paired.append(dict(scenario=sc,comparison='queue_guard minus '+pol,n=N,mean_difference=m,ci95_low=m-1.96*se,ci95_high=m+1.96*se))
write('f8_summary.csv',summary);write('f8_paired.csv',paired)
# Deterministic regression tests for delayed information and rollback presence
assert len(rows)==len(SC)*N*len(POL)
assert all(math.isfinite(r['net']) for r in rows)
assert all(r['rollback']==0 for r in rows if r['policy']=='no_rollback')
assert all(r['rollback']==0 for r in rows if r['policy']=='matched_fixed')
assert all(r['observed_n']<=H*B for r in rows)
assert any(r['rollback'] for r in rows if r['policy']=='queue_guard')
checks=dict(n_rows=len(rows),n_scenarios=len(SC),n_policies=len(POL),n_seeds=N,all_finite=True,rollback_trigger_exercised=True,queue_observation_rule='Only reports with due<=current time are observed; deployment reports may remain pending',caveats=['Simulation is independent of F4 oracle and is NOT a mathematically optimal policy comparison','The no_queue ablation is intentionally information-leaking and is NOT a deployable comparator','Policies use fixed heuristic thresholds; differences do not establish framework novelty','No multiple-testing correction; normal approximate intervals are exploratory','Real SECOM has no intervention trajectories, so cannot validate these counterfactuals','Quality threshold and economics are illustrative, not factory-calibrated'])
(P/'verification.json').write_text(json.dumps(checks,indent=2))
print('ROWS',len(rows));print('SUMMARY');[print(x) for x in summary];print('PAIRED');[print(x) for x in paired if x['scenario'] in ('nominal','late_shift','delayed_reports')]
