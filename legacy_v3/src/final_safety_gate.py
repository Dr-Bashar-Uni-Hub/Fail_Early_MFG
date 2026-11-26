import csv,json,math,random,statistics,functools
from pathlib import Path
from scipy.stats import beta as beta_dist
P=Path(__file__).parent
H=7; ALPHA=1; BETA=9; COST_PILOT=.35; COST_DEPLOY=1.2; REWARD=4.; LOSS=16.; COST_ROLL=3.; MAX_PILOTS=3
# Quality rule: posterior probability that true defect probability exceeds 0.12 <= 0.10.
# Conservative initial gate prohibits deployment; pilots may collect delayed evidence.
LIMIT=.12; TAIL=.10

def bb(n,k,a,b):
 return math.comb(n,k)*math.exp(math.lgamma(a+k)+math.lgamma(b+n-k)-math.lgamma(a+b+n)-math.lgamma(a)-math.lgamma(b)+math.lgamma(a+b))
def planner(lag, constrained, rollback_cost=COST_ROLL):
 @functools.lru_cache(None)
 def pre(t,mode,a,b,q,n):
  if t>=H:return 0.
  k=sum(d==t for d in q);q2=tuple(d for d in q if d>t)
  return sum(bb(k,j,a,b)*decision(t,mode,a+j,b+k-j,q2,n)[1] for j in range(k+1))
 @functools.lru_cache(None)
 def decision(t,mode,a,b,q,n):
  tail=beta_dist.sf(LIMIT,a,b)
  if mode=='pilot':
   opts={'abandon':0.}
   if n<MAX_PILOTS:opts['pilot']=-COST_PILOT+pre(t+1,'pilot',a,b,q+(t+lag,),n+1)
   if not constrained or tail<=TAIL:opts['deploy']=-COST_DEPLOY+pre(t+1,'deployed',a,b,q,n)
  else:
   opts={'rollback':-rollback_cost}
   if not constrained or tail<=TAIL:
    opts['produce']=REWARD-LOSS*a/(a+b)+pre(t+1,'deployed',a,b,q+(t+lag,),n)
  act=max(opts,key=opts.get)
  return act,opts[act]
 return decision,pre

# Independent one-step transition probability normalization checks; exhaustive oracle for tiny horizon in a separate recursion.
def brute(h,lag,constrained):
 def pre(t,mode,a,b,q,n):
  if t>=h:return 0.
  k=sum(d==t for d in q);q2=tuple(d for d in q if d>t)
  return sum(bb(k,j,a,b)*dec(t,mode,a+j,b+k-j,q2,n) for j in range(k+1))
 def dec(t,mode,a,b,q,n):
  tail=beta_dist.sf(LIMIT,a,b)
  if mode=='pilot':
   v=[0.]
   if n<2:v.append(-COST_PILOT+pre(t+1,'pilot',a,b,q+(t+lag,),n+1))
   if not constrained or tail<=TAIL:v.append(-COST_DEPLOY+pre(t+1,'deployed',a,b,q,n))
  else:
   v=[-COST_ROLL]
   if not constrained or tail<=TAIL:v.append(REWARD-LOSS*a/(a+b)+pre(t+1,'deployed',a,b,q+(t+lag,),n))
  return max(v)
 return pre(0,'pilot',ALPHA,BETA,(),0)
checks=[]
for lag in (1,2):
 for constrained in (False,True):
  for h in (3,4):
   # separate reduced DP matched to brute horizon by replacing H through local independent implementation
   def small_dp():
    @functools.lru_cache(None)
    def pre(t,mode,a,b,q,n):
     if t>=h:return 0.
     k=sum(d==t for d in q);r=tuple(d for d in q if d>t)
     return sum(bb(k,j,a,b)*dec(t,mode,a+j,b+k-j,r,n) for j in range(k+1))
    @functools.lru_cache(None)
    def dec(t,mode,a,b,q,n):
     safe=beta_dist.sf(LIMIT,a,b)<=TAIL
     if mode=='pilot':
      vals=[0.]
      if n<2:vals.append(-COST_PILOT+pre(t+1,'pilot',a,b,q+(t+lag,),n+1))
      if not constrained or safe:vals.append(-COST_DEPLOY+pre(t+1,'deployed',a,b,q,n))
     else:
      vals=[-COST_ROLL]
      if not constrained or safe:vals.append(REWARD-LOSS*a/(a+b)+pre(t+1,'deployed',a,b,q+(t+lag,),n))
     return max(vals)
    return pre(0,'pilot',ALPHA,BETA,(),0)
   v=small_dp();z=brute(h,lag,constrained)
   checks.append(dict(horizon=h,lag=lag,constrained=constrained,dp=v,exhaustive=z,abs_error=abs(v-z)))
assert max(r['abs_error'] for r in checks)<1e-9

def simulate(policy,lag,seed,p,rollback_cost):
 rng=random.Random(seed);outcomes=[int(rng.random()<p) for _ in range(H)]
 decision,_=planner_cache[(lag,policy,rollback_cost)] if policy in ('economic','constrained') else (None,None)
 mode='pilot';a,b=ALPHA,BETA;q=[];n=0;net=0.;produced=0;defects=0;unsafe=0;rolled=0;deployed=0;gate_violations=0
 for t in range(H):
  arrived=[v for due,v in q if due==t];q=[(due,v) for due,v in q if due!=t]
  a+=sum(arrived);b+=len(arrived)-sum(arrived)
  if mode=='off':break
  tail=float(beta_dist.sf(LIMIT,a,b))
  if policy in ('economic','constrained'):
   act=decision(t,mode,a,b,tuple(due for due,_ in q),n)[0]
  elif policy=='fixed':
   if mode=='pilot':act='pilot' if n<2 else ('deploy' if tail<=TAIL else 'abandon')
   else:act='produce' if tail<=TAIL else 'rollback'
  else:raise ValueError(policy)
  if act=='abandon':mode='off'
  elif act=='pilot':net-=COST_PILOT;n+=1;q.append((t+lag,outcomes[t]))
  elif act=='deploy':net-=COST_DEPLOY;deployed=1;mode='deployed';gate_violations+=int(tail>TAIL)
  elif act=='produce':net+=REWARD-LOSS*outcomes[t];produced+=1;defects+=outcomes[t];unsafe+=int(p>LIMIT);gate_violations+=int(tail>TAIL);q.append((t+lag,outcomes[t]))
  elif act=='rollback':net-=rollback_cost;rolled=1;mode='off'
 return dict(policy=policy,lag=lag,scenario=p,seed=seed,rollback_cost=rollback_cost,net=net,produced=produced,defects=defects,unsafe_production=unsafe,rollback=rolled,deployed=deployed,posterior_gate_violations=gate_violations)

planner_cache={(lag,policy,c):planner(lag,policy=='constrained',c) for lag in (1,2,3) for policy in ('economic','constrained') for c in (1.,3.,9.)}
rows=[]
for lag in (1,2,3):
 for p in (.035,.10,.25,.40):
  for cost in (1.,3.,9.):
   for seed in range(120):
    for policy in ('economic','constrained','fixed'):
     rows.append(simulate(policy,lag,81000+seed,p,cost))

def write(name,data):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('oracle_checks.csv',checks);write('trajectories.csv',rows)
summary=[];paired=[]
for lag in (1,2,3):
 for p in (.035,.10,.25,.40):
  for cost in (1.,3.,9.):
   subset=[r for r in rows if r['lag']==lag and r['scenario']==p and r['rollback_cost']==cost]
   for pol in ('economic','constrained','fixed'):
    ss=[r for r in subset if r['policy']==pol]
    summary.append(dict(lag=lag,scenario=p,rollback_cost=cost,policy=pol,n=len(ss),mean_net=statistics.mean(r['net'] for r in ss),mean_defects=statistics.mean(r['defects'] for r in ss),mean_unsafe_production=statistics.mean(r['unsafe_production'] for r in ss),deploy_rate=statistics.mean(r['deployed'] for r in ss),rollback_rate=statistics.mean(r['rollback'] for r in ss),mean_gate_violations=statistics.mean(r['posterior_gate_violations'] for r in ss)))
   by={(r['seed'],r['policy']):r for r in subset}
   for comparator in ('economic','fixed'):
    ds=[by[(81000+i,'constrained')]['net']-by[(81000+i,comparator)]['net'] for i in range(120)]
    se=statistics.stdev(ds)/math.sqrt(len(ds))
    paired.append(dict(lag=lag,scenario=p,rollback_cost=cost,comparison='constrained_minus_'+comparator,mean_net_difference=statistics.mean(ds),ci95_low=statistics.mean(ds)-1.96*se,ci95_high=statistics.mean(ds)+1.96*se))
write('summary.csv',summary);write('paired.csv',paired)
(P/'verification.json').write_text(json.dumps({'trajectories':len(rows),'summary_rows':len(summary),'paired_rows':len(paired),'oracle_checks':len(checks),'max_oracle_error':max(r['abs_error'] for r in checks),'no_gate_violations_for_constrained':all(r['posterior_gate_violations']==0 for r in rows if r['policy']=='constrained'),'method':'posterior chance gate, not a guarantee on the true manufacturing defect rate','limitations':['Hard posterior chance constraint can yield no feasible deployments and is not an unconditional real-world safety guarantee','Stationary beta-Bernoulli model, no causal interventions','Fixed comparator is a heuristic, not a full-horizon safety-constrained optimal policy','One-step outcome streams paired, but pilot and production use same time-indexed latent draws','Model validation uses reduced exact recursion only','No industrial treatment outcome data','Exploratory confidence intervals unadjusted']},indent=2))
print('rows',len(rows),'oracle checks',len(checks))
for x in summary:
 if x['lag']==2 and x['rollback_cost']==3:print(x)
