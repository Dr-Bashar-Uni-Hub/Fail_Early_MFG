import json,csv,random,statistics,math,pathlib
P=pathlib.Path(__file__).parent
# All costs are synthetic, normalized monetary units; no real industrial claims.
CONFIG={'seed':20261122,'replications':400,'horizon':24,'pilot_duration':4,'prior_alpha':2,'prior_beta':18,'quality_limit':0.12,'posterior_risk_limit':0.10,'pilot_cost_per_period':0.8,'benefit_per_period':3.0,'defect_loss':24.0,'rollback_cost':4.0,'deploy_cost':3.0,'test_sample_size':25,'candidate_defect_rates':[0.025,0.06,0.10,0.16,0.24], 'baseline_defect_rate':0.10}
(P/'config.json').write_text(json.dumps(CONFIG,indent=2))
# Exact posterior tail P(p > threshold) for integer beta shapes via binomial identity
from math import comb
def beta_tail(a,b,x):
    n=a+b-1
    cdf=sum(comb(n,k)*x**k*(1-x)**(n-k) for k in range(a,n+1))
    return 1-cdf

def posterior_mean(s,n):return (2+s)/(20+n)
def expected_increment(p,h):return h*(3.0-24.0*p)-3.0

def run_one(seed,p,policy):
    rng=random.Random(seed)
    # Common random numbers for all policies, including counterfactual production stream.
    draws=[rng.random() for _ in range(24*25+100)]
    # Pilot observes 25 independent parts per pilot period; production is normalized to 1 period of value.
    def obs(period):return sum(draws[period*25+j]<p for j in range(25))
    def outcome(period):return 3.0-24.0*(obs(period)/25)
    total=0.; samples=0; decision='abandon'; pilot_periods=0
    if policy=='never':return dict(value=0.,decision='abandon',pilot_periods=0,unsafe_deploy=0)
    if policy=='always':
        total-=3.0
        for t in range(24):total+=outcome(t)
        return dict(value=total,decision='deploy',pilot_periods=0,unsafe_deploy=int(p>0.12))
    # Fixed pilot: 4 periods then empirical threshold; Bayesian: optional stop at each period.
    for t in range(4):
        total-=0.8
        samples+=obs(t);pilot_periods+=1
        if policy=='bayes' and t>=1:
            # Bayesian decision with uncertainty gate; abandon early if mean economics negative and no plausible recovery.
            if posterior_mean(samples,(t+1)*25)>0.19:break
    n=pilot_periods*25
    if policy=='fixed':
        deploy=(samples/n)<=0.12
    elif policy=='bayes':
        deploy=(beta_tail(2+samples,18+n-samples,0.12)<=0.10 and expected_increment(posterior_mean(samples,n),24-pilot_periods)>0)
    else:raise ValueError(policy)
    if deploy:
        total-=3.0
        for t in range(pilot_periods,24):total+=outcome(t)
        decision='deploy'
    else:
        # Rollback cost applies to tested process only in this stylized setting.
        total-=4.0
        decision='rollback'
    return dict(value=total,decision=decision,pilot_periods=pilot_periods,unsafe_deploy=int(deploy and p>0.12))

rows=[]
for p in CONFIG['candidate_defect_rates']:
 for r in range(CONFIG['replications']):
  for policy in ['never','always','fixed','bayes']:
   out=run_one(CONFIG['seed']+r*7919+int(p*100000),p,policy)
   rows.append(dict(defect_rate=p,replication=r,policy=policy,**out))
with (P/'raw_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary=[]
for p in CONFIG['candidate_defect_rates']:
 for policy in ['never','always','fixed','bayes']:
  subset=[r for r in rows if r['defect_rate']==p and r['policy']==policy]
  vals=[r['value'] for r in subset]
  summary.append(dict(defect_rate=p,policy=policy,mean_net_value=round(statistics.mean(vals),4),sd_net_value=round(statistics.stdev(vals),4),deployment_rate=round(sum(r['decision']=='deploy' for r in subset)/len(subset),4),unsafe_deployment_rate=round(statistics.mean(r['unsafe_deploy'] for r in subset),4),mean_pilot_periods=round(statistics.mean(r['pilot_periods'] for r in subset),4)))
with (P/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
# Verification of exact beta-binomial tail for a=2,b=18, x=.12 against simple numerical integration.
def integrate_beta_tail(a,b,x,steps=20000):
 norm=math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
 dx=(1-x)/steps
 return sum(norm*(x+(j+.5)*dx)**(a-1)*(1-x-(j+.5)*dx)**(b-1)*dx for j in range(steps))
checks=[]
for a,b in [(2,18),(4,20),(10,30),(20,50)]:
 err=abs(beta_tail(a,b,.12)-integrate_beta_tail(a,b,.12))
 checks.append({'a':a,'b':b,'error':err})
assert max(c['error'] for c in checks)<1e-6,checks
assert len(rows)==5*400*4
(P/'verification.json').write_text(json.dumps({'rows':len(rows),'beta_tail_checks':checks,'passed':True},indent=2))
print('Rows',len(rows),'checks',checks)
for s in summary:print(s)
