import csv,json,random,math,statistics
from pathlib import Path
from scipy.stats import beta
P=Path(__file__).parent
LIMIT=.12; TAIL=.10; H=30; LAG=2; ALPHA=1; BETA=9
COST_PILOT=.35; COST_DEPLOY=1.2; REWARD=4.; LOSS=16.; ROLLBACK=3.
POLICIES=('abandon','economic_immediate','pilot_3','pilot_10','pilot_14')
PROBS=(.035,.10,.25,.40)
NSEEDS=1000

def simulate(p,seed,policy):
 rng=random.Random(seed); outcomes=[int(rng.random()<p) for _ in range(H)]
 a,b=ALPHA,BETA; pending=[]; net=0.; deployed=False; rolled=False; pilots=0; produced=0; defects=0; unsafe=0; violations=0; deployment_time=-1
 pilot_target={'pilot_3':3,'pilot_10':10,'pilot_14':14}.get(policy,0)
 for t in range(H):
  arrived=[x for due,x in pending if due==t];pending=[(due,x) for due,x in pending if due!=t]
  a+=sum(arrived);b+=len(arrived)-sum(arrived)
  safe=beta.sf(LIMIT,a,b)<=TAIL
  if policy=='abandon':break
  if policy=='economic_immediate':
   if not deployed:
    deployed=True;net-=COST_DEPLOY;deployment_time=t
   else:
    net+=REWARD-LOSS*outcomes[t];produced+=1;defects+=outcomes[t];unsafe+=int(p>LIMIT);violations+=int(not safe);pending.append((t+LAG,outcomes[t]))
   continue
  if not deployed:
   if pilots<pilot_target:
    pilots+=1;net-=COST_PILOT;pending.append((t+LAG,outcomes[t]));continue
   if pending:continue
   if safe:
    deployed=True;net-=COST_DEPLOY;deployment_time=t
   break_if_not_safe=not safe
   if break_if_not_safe:break
  else:
   if not safe:
    net-=ROLLBACK;rolled=True;break
   net+=REWARD-LOSS*outcomes[t];produced+=1;defects+=outcomes[t];unsafe+=int(p>LIMIT);pending.append((t+LAG,outcomes[t]))
 return dict(true_p=p,seed=seed,policy=policy,net=net,produced=produced,defects=defects,unsafe_units=unsafe,pilots=pilots,deployed=int(deployed),rollback=int(rolled),gate_violations=violations,deployment_time=deployment_time)

rows=[]
for p in PROBS:
 for seed in range(NSEEDS):
  for pol in POLICIES:rows.append(simulate(p,110000+seed,pol))

def write(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
write('trajectories.csv',rows)
summary=[]
for p in PROBS:
 for pol in POLICIES:
  r=[x for x in rows if x['true_p']==p and x['policy']==pol]
  summary.append(dict(true_p=p,policy=pol,n=len(r),mean_net=statistics.mean(x['net'] for x in r),mean_throughput=statistics.mean(x['produced'] for x in r),mean_defects=statistics.mean(x['defects'] for x in r),mean_unsafe_units=statistics.mean(x['unsafe_units'] for x in r),deployment_rate=statistics.mean(x['deployed'] for x in r),rollback_rate=statistics.mean(x['rollback'] for x in r),mean_gate_violations=statistics.mean(x['gate_violations'] for x in r)))
write('summary.csv',summary)
# Independent algebraic feasibility: zero failures and total observation count, no policy simulation involved.
feas=[]
for prior_a,prior_b in ((1,9),(1,19),(.5,9.5)):
 for limit,tail in ((.12,.1),(.12,.2),(.15,.1),(.2,.1)):
  n=next(n for n in range(200) if beta.sf(limit,prior_a,prior_b+n)<=tail)
  feas.append(dict(prior_a=prior_a,prior_b=prior_b,defect_limit=limit,tail_limit=tail,minimum_zero_failure_samples=n,initial_tail_probability=beta.sf(limit,prior_a,prior_b)))
write('feasibility.csv',feas)
# paired bootstrap using deterministic resampling
pairs=[]
for p in PROBS:
  by={(x['seed'],x['policy']):x for x in rows if x['true_p']==p}
  for pol in ('pilot_3','pilot_10','pilot_14'):
   ds=[by[(110000+i,pol)]['net']-by[(110000+i,'abandon')]['net'] for i in range(NSEEDS)]
   rng=random.Random(4242)
   boots=[statistics.mean(ds[rng.randrange(NSEEDS)] for _ in range(NSEEDS)) for _ in range(500)]
   boots.sort()
   pairs.append(dict(true_p=p,comparison=pol+'_minus_abandon',mean_difference=statistics.mean(ds),bootstrap95_low=boots[12],bootstrap95_high=boots[487]))
write('paired.csv',pairs)
assert len(rows)==20000
assert all(x['gate_violations']==0 for x in rows if x['policy'] not in ('economic_immediate',))
assert all(x['produced']==0 for x in rows if x['policy']=='pilot_3')
assert all(x['produced']<=H for x in rows)
(P/'verification.json').write_text(json.dumps(dict(trajectory_count=len(rows),summary_count=len(summary),feasibility_count=len(feas),paired_count=len(pairs),all_safety_policies_respect_posterior_gate=True,pilot_3_production_zero=True,simulation_model='stationary independent Bernoulli, paired per-period outcomes; pilot and production both observe same process',limitations=['Fixed pilot policies are heuristics, not exact Bayesian optimizers','Pilot costs and time horizon are assumed, not measured','Posterior gate is not physical safety guarantee','Pilot and production outcomes are stationary; no causal interventions','A policy can satisfy gate yet have unsafe true defect probability','No matched full-horizon safety-constrained comparator for expanded horizon','Bootstrap intervals exploratory']),indent=2))
for x in summary:print(x)
