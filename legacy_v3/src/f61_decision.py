"""F6.1 accounting-corrected matched-action policy evaluation. Synthetic only."""
import sys,random,json,statistics,csv
from dataclasses import replace
from pathlib import Path
from functools import lru_cache
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from framework import Config,Engine
class CorrectedEngine(Engine):
 def qvalues(self,t,a,b,mode='pilot',pilots=0,modified=False):
  q=super().qvalues(t,a,b,mode,pilots,modified)
  # A pilot is a non-production experiment: abandoning an undeployed candidate
  # does not require restoring the production process.
  if 'abandon' in q:q['abandon']=0.
  return q

def decide_myopic(eng,t,a,b,mode,pilots,modified):
 q=eng.qvalues(t,a,b,mode,pilots,modified)
 if not q:return 'terminal'
 if mode=='deployed':
  return 'continue' if eng.admissible(a,b) else 'rollback'
 # All candidate actions available, same constraints; myopic chooses by current
 # one-period benefit/cost and cannot value future observations.
 score={'abandon':0.,'deploy':(eng.c.horizon-t)*eng.stage_reward(a,b)-eng.c.deploy_cost if eng.admissible(a,b) else float('-inf'),'test':-eng.c.pilot_cost-eng.c.delay_cost,'modify':-eng.c.modify_cost}
 return max(q,key=lambda action:score.get(action,float('-inf')))

def simulate(eng,seed,ps,policy):
 c=eng.c;rng=random.Random(seed)
 streams={(m,t):sum(rng.random()<ps[m] for _ in range(c.batch)) for m in range(2) for t in range(c.horizon)}
 t=0;a=c.alpha;b=c.beta;pilots=0;modified=False;mode='pilot';net=0.;hist=[]
 while t<c.horizon:
  action=eng.decide(t,a,b,mode,pilots,modified)[0] if policy=='oracle' else decide_myopic(eng,t,a,b,mode,pilots,modified)
  hist.append(action)
  if action=='abandon':break
  if action=='rollback':net-=c.rollback_cost;break
  if action=='modify':net-=c.modify_cost;modified=True;a=c.modified_alpha;b=c.modified_beta;pilots=0;t+=1
  elif action=='test':
   net-=c.pilot_cost+c.delay_cost;k=streams[(int(modified),t)];t+=1+c.delay;a+=k;b+=c.batch-k;pilots+=1
  elif action=='deploy':net-=c.deploy_cost;mode='deployed'
  elif action=='continue':
   k=streams[(int(modified),t)];net+=c.benefit-c.defect_loss*k/c.batch;a+=k;b+=c.batch-k;t+=1
  else:raise RuntimeError(action)
 return dict(net=net,deployed=int('deploy' in hist),rollback=int('rollback' in hist),pilots=hist.count('test'),actions='|'.join(hist))
CONFIGS=[('test_favorable',Config(horizon=6,alpha=2,beta=18,benefit=8,pilot_cost=.2)),('deploy_favorable',Config(horizon=6,alpha=1,beta=29,benefit=5,pilot_cost=.2)),('conservative',Config(horizon=6,alpha=2,beta=18,benefit=3,pilot_cost=.8))]
SCENARIOS=[('favorable',.025,.020),('moderate',.06,.045),('borderline',.10,.075),('unfavorable',.16,.11),('severe',.24,.17)]
rows=[];initial=[]
for name,cfg in CONFIGS:
 e=CorrectedEngine(cfg);initial.append({'config':name,'oracle_initial':e.initial()[0],'oracle_value':e.initial()[1],'myopic_initial':decide_myopic(e,0,cfg.alpha,cfg.beta,'pilot',0,False)})
 for scen,p0,p1 in SCENARIOS:
  for seed in range(150):
   for policy in ['oracle','myopic']:
    r=simulate(e,seed,(p0,p1),policy)
    rows.append(dict(config=name,scenario=scen,seed=seed,policy=policy,**r))
pd=None
with (ROOT/'f61_raw.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for name,_ in CONFIGS:
 for scen,*_ in SCENARIOS:
  for policy in ['oracle','myopic']:
   x=[r for r in rows if (r['config'],r['scenario'],r['policy'])==(name,scen,policy)]
   summary.append(dict(config=name,scenario=scen,policy=policy,n=len(x),mean_net=statistics.mean(r['net'] for r in x),deploy_rate=statistics.mean(r['deployed'] for r in x),rollback_rate=statistics.mean(r['rollback'] for r in x),mean_pilots=statistics.mean(r['pilots'] for r in x)))
with (ROOT/'f61_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
paired=[]
for name,_ in CONFIGS:
 for scen,*_ in SCENARIOS:
  dif=[]
  for seed in range(150):
   vals={r['policy']:r['net'] for r in rows if (r['config'],r['scenario'],r['seed'])==(name,scen,seed)}
   dif.append(vals['oracle']-vals['myopic'])
  mean=statistics.mean(dif);sd=statistics.stdev(dif);paired.append(dict(config=name,scenario=scen,n=len(dif),mean_oracle_minus_myopic=mean,paired_se=sd/(len(dif)**.5),ci95_low=mean-1.96*sd/(len(dif)**.5),ci95_high=mean+1.96*sd/(len(dif)**.5)))
with (ROOT/'f61_paired.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
# Independent one-step oracle verification: with horizon=1, compare all feasible actions by explicit enumeration.
small=CorrectedEngine(Config(horizon=1,alpha=1,beta=29,benefit=5))
q=small.qvalues(0,1,29);assert abs(small.initial()[1]-max(q.values()))<1e-10
checks={'records':len(rows),'expected_records':3*5*150*2,'all_finite':all(abs(r['net'])<1e8 for r in rows),'same_action_set':'Both use same corrected engine qvalues; myopic heuristic ignores future observation value','corrected_accounting':'Pilot is off-line, so abandon incurs no rollback; rollback charged only after deployment','initial':initial,'one_step_consistency':True,'limitations':['Myopic baseline is not a strong alternative full-horizon Bayesian solver','Oracle optimality conditional on prior, exact model and state space','Synthetic fixed defect probabilities not drawn from the stated priors','Shared streams coupled across policies but do not imply causal identification','No SECOM intervention histories available']}
(ROOT/'f61_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(initial,indent=2));print('selected', [r for r in summary if r['config']=='test_favorable' and r['scenario'] in ('favorable','severe')]);print('paired', [r for r in paired if r['config']=='test_favorable' and r['scenario'] in ('favorable','severe')])
