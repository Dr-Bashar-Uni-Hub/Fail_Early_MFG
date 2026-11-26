import csv,math,statistics,json,random,functools
from pathlib import Path
P=Path(__file__).parent
src=Path('/mnt/data/FAIL_EARLY_MFG_F7_1_F8/f71_f8_raw.csv')
rows=list(csv.DictReader(src.open()))
for r in rows:
 for k in ('net','pilots','deployed','rollback','defects','produced','unsafe_periods','pending_end','observed_n'):r[k]=float(r[k])
pols=sorted(set(r['policy'] for r in rows));scs=sorted(set(r['scenario'] for r in rows))
# Independent recomputation of paired metrics, safety and a preregistered-like constraint screen.
out=[]
for sc in scs:
 for pol in pols:
  rr=[r for r in rows if r['scenario']==sc and r['policy']==pol]
  out.append(dict(scenario=sc,policy=pol,n=len(rr),mean_net=statistics.mean(x['net'] for x in rr),mean_unsafe_periods=statistics.mean(x['unsafe_periods'] for x in rr),mean_defective_units=statistics.mean(x['defects'] for x in rr),mean_produced_units=statistics.mean(x['produced'] for x in rr),rollback_rate=statistics.mean(x['rollback'] for x in rr),deploy_rate=statistics.mean(x['deployed'] for x in rr),meets_unsafe_budget_2=statistics.mean(x['unsafe_periods'] for x in rr)<=2))
def write(name,rr):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
write('f9_independent_reaggregation.csv',out)
# matched outcome audit: same seed and scenario, but action sets and information policies not equivalent.
# Fairness flag for each policy pair, not based on performance.
fair=[]
for pol in pols:
 fair.append(dict(policy=pol,common_random_outcomes=True,same_reporting_delay=(pol!='no_queue'),same_pilot_and_deploy_cost=True,same_rollback_action=(pol not in ('matched_fixed','no_rollback','no_risk_gate')),same_deployment_gate=(pol in ('queue_guard','no_rollback','no_queue')),full_horizon_optimal=False,valid_superiority_comparator=(pol=='matched_reactive'),notes='matched_reactive has different stopping rules, so not identical feasible decision class' if pol=='matched_reactive' else ('information leakage; invalid real-world baseline' if pol=='no_queue' else 'not full-horizon matched Bayesian comparator')))
write('f9_comparator_fairness.csv',fair)
# Independent exact Bellman DP in deliberately small, fully specified binomial model. Also exhaustive policy-tree enumeration by recursion.
# 3 decisions max; choose abandon, deploy (terminal), or test (one binomial sample, consume a period).
# prior Beta(a,b), batch n=2, true defect p random prior, risk-neutral reward net = 10 - 25*p, test cost 0.35.
from math import comb
N=2; COST=.35

def transitions(a,b):
 den=math.prod(range(a+b,a+b+N))/math.prod(range(1,N+1))
 # beta-binomial P(k)=C(N,k)*(a)_k*(b)_(N-k)/(a+b)_N
 for k in range(N+1):
  num=comb(N,k)*math.prod(range(a,a+k))*math.prod(range(b,b+N-k))
  yield k,num/math.prod(range(a+b,a+b+N))
def deploy_value(a,b):return 10-25*a/(a+b)
@functools.lru_cache(None)
def dp(a,b,h):
 acts={'abandon':0.,'deploy':deploy_value(a,b)}
 if h>0:acts['test']=-COST+sum(pk*dp(a+k,b+N-k,h-1)[0] for k,pk in transitions(a,b))
 action=max(acts,key=acts.get)
 return acts[action],action
# separate enumerator over all deterministic decision trees (value set), independently checks Bellman maximization
@functools.lru_cache(None)
def all_values(a,b,h):
 vals=[(0.,'abandon'),(deploy_value(a,b),'deploy')]
 if h>0:
  branches=[list(all_values(a+k,b+N-k,h-1)) for k in range(N+1)]
  import itertools
  for combination in itertools.product(*branches):
   v=-COST+sum(pk*combination[k][0] for k,pk in transitions(a,b))
   vals.append((v,'test'))
 return tuple(vals)
checks=[]
for a,b in ((1,4),(2,8),(1,12)):
 for h in (0,1,2):
  d,act=dp(a,b,h); allv=all_values(a,b,h); brute=max(x[0] for x in allv)
  checks.append(dict(alpha=a,beta=b,horizon=h,dp_value=d,exhaustive_value=brute,absolute_error=abs(d-brute),root_action=act,enumerated_policy_trees=len(allv)))
write('f9_exact_oracle_unit_tests.csv',checks)
assert all(x['absolute_error']<1e-10 for x in checks)
# paired bootstrap confidence intervals for main policy contrasts (resampling paired seeds)
rng=random.Random(20261122)
paired=[]
for sc in scs:
 for comp in ('matched_fixed','matched_reactive'):
  by={(r['seed'],r['policy']):r for r in rows if r['scenario']==sc}
  seeds=sorted(set(r['seed'] for r in rows if r['scenario']==sc))
  diff=[by[(seed,'queue_guard')]['net']-by[(seed,comp)]['net'] for seed in seeds]
  boot=[statistics.mean(diff[rng.randrange(len(diff))] for _ in diff) for __ in range(1000)]
  boot.sort(); paired.append(dict(scenario=sc,comparison='queue_guard minus '+comp,n=len(diff),mean_difference=statistics.mean(diff),bootstrap_95_low=boot[24],bootstrap_95_high=boot[974]))
write('f9_paired_bootstrap.csv',paired)
# checks for reporting and costs: audit source structure, identify invalid inference.
verification=dict(records=len(rows),expected=6000,independent_reaggregation_rows=len(out),exact_small_model_tests=len(checks),max_exact_error=max(x['absolute_error'] for x in checks),paired_bootstrap_rows=len(paired),critical_findings=['Exact Bellman recursion verified against exhaustive enumeration ONLY for a small standalone beta-binomial toy model; this does not verify F4 or F7.1 full-horizon implementation','F7.1 queue_guard is a heuristic, not a full-horizon Bayesian policy','No full-horizon matched-information comparator is implemented in F7.1','Unsafe period budget is a post-hoc descriptive screen, not an ex ante chance constraint','Reported 200 paired seeds are synthetic; SECOM does not provide interventions or causal policy evaluation','F7.1 labels unsafe periods using latent true p, not observed defect counts','Cost parameters illustrative; factory-calibrated utility unavailable'])
assert len(rows)==6000
(P/'verification.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification,indent=2));print('PAIRED');[print(x) for x in paired]
