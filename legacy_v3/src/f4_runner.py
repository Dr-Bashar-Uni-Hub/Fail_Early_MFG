import csv,json,statistics,math,random
from pathlib import Path
from framework import Config,Engine,simulate
P=Path(__file__).parent
c=Config();e=Engine(c)
scenarios=[(.025,.04),(.06,.07),(.10,.08),(.16,.10),(.24,.14)]
policies=['oracle','fixed','risk_gate']
rows=[]
for j,ps in enumerate(scenarios):
 for r in range(120):
  for policy in policies:
   o=simulate(e,20261122+j*100000+r,ps,policy)
   rows.append(dict(scenario=j,original_p=ps[0],modified_p=ps[1],replication=r,policy=policy,
                    net_value=o['net_value'],deployed=int(o['deployed']),modified=int(o['modified']),
                    rollback=int(o['rollback']),pilot_count=o['pilot_count']))
with (P/'raw_results_f4.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
summary=[]
for j in range(len(scenarios)):
 for policy in policies:
  x=[r for r in rows if r['scenario']==j and r['policy']==policy]
  values=[r['net_value'] for r in x]
  summary.append(dict(scenario=j,original_p=scenarios[j][0],modified_p=scenarios[j][1],policy=policy,
    mean_net_value=round(statistics.mean(values),4),sd_net_value=round(statistics.stdev(values),4),
    deployment_rate=round(statistics.mean(r['deployed'] for r in x),4),
    rollback_rate=round(statistics.mean(r['rollback'] for r in x),4),
    modification_rate=round(statistics.mean(r['modified'] for r in x),4),
    mean_pilot_count=round(statistics.mean(r['pilot_count'] for r in x),4)))
with (P/'summary_f4.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,summary[0].keys());w.writeheader();w.writerows(summary)
# Numerical invariants and independent beta-binomial predictive normalization
checks={'predictive_pmf_normalized':all(abs(sum(e.pmf(a,b,c.batch))-1)<1e-12 for a,b in [(2,18),(5,23),(9,29)]),
 'zero_horizon':e._value(c.horizon,c.alpha,c.beta,'pilot',0,False)==0,
 'oracle_equals_max_actions':abs(e._value(0,c.alpha,c.beta,'pilot',0,False)-max(e.qvalues(0,c.alpha,c.beta).values()))<1e-12,
 'all_values_finite':all(math.isfinite(r['net_value']) for r in rows),
 'row_count':len(rows)==len(scenarios)*120*len(policies),
 'delay_is_one_period':c.delay==1}
assert all(checks.values()),checks
(P/'verification_f4.json').write_text(json.dumps({'checks':checks,'initial_decision':e.initial(),
 'cache_info':str(e._value.cache_info()),'config':vars(c)},indent=2))
print('initial',e.initial(),'states',e._value.cache_info())
for x in summary:print(x)
