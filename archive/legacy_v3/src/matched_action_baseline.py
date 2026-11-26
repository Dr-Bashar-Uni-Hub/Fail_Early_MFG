"""Exact matched-action reduced-horizon delayed-feedback benchmark.
Both optimal and heuristic policies use identical action sets, reporting delays,
observation histories and costs. The full legacy 12-period simulator is NOT solved.
"""
import csv, json, math, random, argparse
from pathlib import Path
from functools import lru_cache
from statistics import mean

THETA=(0.035,0.24); PRIOR=(0.7,0.3)
H=8; B=3; DELAY=2; MAX_PILOTS=2
PILOT_COST=0.8; DEPLOY_COST=2.0; ROLLBACK_COST=4.0; REVENUE=5.0; DEFECT_PENALTY=18.0

def posterior(g,b):
    weights=[PRIOR[i]*THETA[i]**b*(1-THETA[i])**g for i in range(2)]
    s=sum(weights)
    return weights[1]/s

def predict_k(g,b,k):
    q=posterior(g,b)
    return math.comb(B,k)*((1-q)*THETA[0]**k*(1-THETA[0])**(B-k)+q*THETA[1]**k*(1-THETA[1])**(B-k))

def actions(s):
    t,mode,g,b,due,pilots=s
    if t>=H:return ()
    if due>=0:return ('wait',)
    if mode=='pilot':return ('abandon',)+(('pilot',) if pilots<MAX_PILOTS and t+DELAY<H else ())+('deploy',)
    return ('rollback','produce')

def transition(s,a):
    t,mode,g,b,due,pilots=s
    if a in ('abandon','rollback'):return (0.0,[])
    if a=='deploy':return (-DEPLOY_COST,[(1.0,(t+1,'deployed',g,b,-1,pilots))])
    if a=='pilot':return (-PILOT_COST,[(1.0,(t+1,mode,g,b,t+DELAY,pilots+1))])
    if a=='produce':
        expected_defect=B*((1-posterior(g,b))*THETA[0]+posterior(g,b)*THETA[1])
        return (REVENUE-DEFECT_PENALTY*expected_defect/B,[(1.0,(t+1,mode,g,b,t+DELAY,pilots))])
    if a=='wait':
        if due>t:return (0.0,[(1.0,(t+1,mode,g,b,due,pilots))])
        return (0.0,[(predict_k(g,b,k),(t+1,mode,g+B-k,b+k,-1,pilots)) for k in range(B+1)])
    raise ValueError(a)

@lru_cache(None)
def V(s):
    aa=actions(s)
    if not aa:return 0.0
    return max(Q(s,a) for a in aa)

def Q(s,a):
    if a=='rollback':return -ROLLBACK_COST
    r,branches=transition(s,a)
    return r+sum(prob*V(ns) for prob,ns in branches)

def choice(s,policy):
    aa=actions(s)
    if not aa:return None
    if policy=='exact':return max(aa,key=lambda a:(Q(s,a),-aa.index(a)))
    if policy=='greedy':
        # Same available actions and information, but no lookahead beyond immediate reward.
        immediate={a: (-ROLLBACK_COST if a=='rollback' else transition(s,a)[0]) for a in aa}
        return max(aa,key=lambda a:(immediate[a],-aa.index(a)))
    if policy=='fixed':
        if 'wait' in aa:return 'wait'
        if s[1]=='pilot':
            if s[-1]<1 and 'pilot' in aa:return 'pilot'
            return 'deploy' if posterior(s[2],s[3])<0.4 else 'abandon'
        return 'produce' if posterior(s[2],s[3])<0.4 else 'rollback'
    raise ValueError(policy)

def simulate(seed,theta,policy):
    rng=random.Random(seed)
    # Common potential observations indexed by decision time, not policy step count.
    potential={t:sum(rng.random()<theta for _ in range(B)) for t in range(H)}
    s=(0,'pilot',0,0,-1,0); pending_k=None; reward=0.; production=0; defects=0; deploy=0; rollback=0; pilot=0
    while s[0]<H:
        a=choice(s,policy)
        if a is None:break
        t,mode,g,b,due,pilots=s
        if a=='abandon':break
        if a=='rollback':reward-=ROLLBACK_COST;rollback+=1;break
        if a=='deploy':reward-=DEPLOY_COST;deploy=1;s=(t+1,'deployed',g,b,-1,pilots)
        elif a in ('pilot','produce'):
            k=potential[t];pending_k=k
            if a=='pilot':reward-=PILOT_COST;pilot+=1
            else:reward+=REVENUE-DEFECT_PENALTY*k/B;production+=B;defects+=k
            s=(t+1,mode,g,b,t+DELAY,pilots+(a=='pilot'))
        else:
            if due>t:s=(t+1,mode,g,b,due,pilots)
            else:
                assert pending_k is not None
                s=(t+1,mode,g+B-pending_k,b+pending_k,-1,pilots);pending_k=None
    return dict(net=reward,production=production,defects=defects,deployed=deploy,rollback=rollback,pilots=pilot)

def brute(s):
    aa=actions(s)
    if not aa:return 0.
    vals=[]
    for a in aa:
        if a=='rollback':vals.append(-ROLLBACK_COST);continue
        r,branches=transition(s,a)
        vals.append(r+sum(prob*brute(ns) for prob,ns in branches))
    return max(vals)

def main(out,n):
    out.mkdir(parents=True,exist_ok=True)
    states=[(0,'pilot',0,0,-1,0),(2,'pilot',3,0,-1,1),(3,'pilot',2,1,-1,1),(4,'deployed',3,0,-1,1),(5,'deployed',3,3,-1,1)]
    checks=[dict(state=str(s),bellman=V(s),enumeration=brute(s),abs_error=abs(V(s)-brute(s))) for s in states]
    assert max(c['abs_error'] for c in checks)<1e-10
    rows=[]
    for theta in THETA:
        for seed in range(n):
            for policy in ('exact','greedy','fixed'):
                rows.append(dict(theta=theta,seed=seed,policy=policy,**simulate(seed,theta,policy)))
    def write(name,rr):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
    write('matched_action_raw.csv',rows);write('oracle_checks.csv',checks)
    summaries=[];paired=[]
    for theta in THETA:
        for pol in ('exact','greedy','fixed'):
            rr=[r for r in rows if r['theta']==theta and r['policy']==pol]
            summaries.append(dict(theta=theta,policy=pol,n=len(rr),mean_net=mean(r['net'] for r in rr),mean_production=mean(r['production'] for r in rr),deploy_rate=mean(r['deployed'] for r in rr),rollback_rate=mean(r['rollback'] for r in rr)))
        for other in ('greedy','fixed'):
            diffs=[next(r['net'] for r in rows if r['theta']==theta and r['seed']==seed and r['policy']=='exact')-next(r['net'] for r in rows if r['theta']==theta and r['seed']==seed and r['policy']==other) for seed in range(n)]
            paired.append(dict(theta=theta,comparison='exact minus '+other,n=n,mean_diff=mean(diffs)))
    write('matched_action_summary.csv',summaries);write('matched_action_paired.csv',paired)
    (out/'verification.json').write_text(json.dumps(dict(oracle_checks=len(checks),max_abs_error=max(c['abs_error'] for c in checks),trajectories=len(rows),seed_count=n,root_exact_action=choice(states[0],'exact'),root_exact_prior_value=V(states[0]),model_scope='Reduced 8-period discrete two-point latent defect model; one outstanding report at a time; same actions for all policies; NOT the original 12-period F7.1 simulator',limitations=['Model is intentionally small; not a full-scale oracle','Greedy/fixed are heuristic baselines; not established state-of-art algorithms','Prior-expected optimality does not guarantee conditional dominance for fixed true defect rates','Reporting queue restricts to one outstanding batch and forbids production while reports pending']),indent=2))
    print('root',choice(states[0],'exact'),V(states[0]));print('checks',checks);print('summaries',summaries)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,default=250);args=ap.parse_args();main(args.output,args.seeds)
