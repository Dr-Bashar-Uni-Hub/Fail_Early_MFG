"""Twelve-period matched-action heuristic comparison. Not a full-horizon Bayesian oracle."""
import argparse,csv,random,math,statistics
from pathlib import Path
SC={'nominal':(.035,.025,1,0),'late_shift':(.035,.24,1,4),'high_defect':(.22,.22,1,0),'delayed_reports':(.035,.025,3,0),'rollback_expensive':(.035,.24,1,4)}
POL=('queue_guard','fixed_rule','reactive_rule')
H=12;B=20

def decide(pol,mode,n,bad,pilots,pending):
    rate=bad/n if n else 1.
    upper=(rate+1.96**2/(2*n)+1.96*math.sqrt(rate*(1-rate)/n+1.96**2/(4*n*n)))/(1+1.96**2/n) if n else 1.
    if mode=='pilot':
        if pol=='queue_guard':return 'deploy' if n>=40 and upper<=.18 else ('abandon' if pilots>=5 and not pending else 'pilot')
        if pol=='fixed_rule':return ('deploy' if rate<=.12 else 'abandon') if pilots>=2 and n>=40 else 'pilot'
        return 'deploy' if n>=40 and upper<=.18 else ('abandon' if pilots>=4 else 'pilot')
    # All three policies are permitted to rollback, continue or abandon; their decision rules differ.
    if pol=='fixed_rule':return 'rollback' if n>=40 and rate>.20 else 'continue'
    if pol=='reactive_rule':return 'rollback' if n>=40 and rate>.12 else 'continue'
    return 'rollback' if n>=40 and rate>.12 else 'continue'

def sim(seed,scenario,policy):
    p0,p1,lag,shift=SC[scenario]; rng=random.Random(seed)
    stream=[[rng.random() for _ in range(B)] for _ in range(H)]
    pending=[];n=bad=pilots=0;mode='pilot';net=0.;prod=defects=unsafe=0;rolled=deployed=0
    actions=[]
    for t in range(H):
        arrived=[x for x in pending if x[0]<=t];pending=[x for x in pending if x[0]>t]
        for due,k in arrived:n+=B;bad+=k
        action=decide(policy,mode,n,bad,pilots,pending)
        actions.append(action)
        if action=='abandon':break
        if action=='rollback':rolled=1;net-=18 if scenario=='rollback_expensive' else 6;break
        if action=='deploy':mode='deployed';deployed=1;net-=3
        p=p1 if shift and t>=shift else p0
        k=sum(u<p for u in stream[t])
        if action=='pilot':pilots+=1;net-=.5
        else:prod+=B;defects+=k;unsafe+=int(p>.12);net+=8-35*k/B
        pending.append((t+lag,k))
    return dict(net=net,produced=prod,defects=defects,unsafe_periods=unsafe,rollback=rolled,deployed=deployed,pilots=pilots,actions='|'.join(actions))

def run(out,seeds):
    out.mkdir(parents=True,exist_ok=True)
    rows=[dict(scenario=sc,seed=seed,policy=pol,**sim(seed,sc,pol)) for sc in SC for seed in range(seeds) for pol in POL]
    def write(name,records):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write('full_matched_raw.csv',rows)
    summary=[];paired=[]
    for sc in SC:
        for pol in POL:
            rs=[r for r in rows if r['scenario']==sc and r['policy']==pol]
            summary.append(dict(scenario=sc,policy=pol,n=len(rs),mean_net=statistics.mean(r['net'] for r in rs),mean_produced=statistics.mean(r['produced'] for r in rs),mean_unsafe=statistics.mean(r['unsafe_periods'] for r in rs),rollback_rate=statistics.mean(r['rollback'] for r in rs)))
        for pol in POL[1:]:
            a={r['seed']:r for r in rows if r['scenario']==sc and r['policy']=='queue_guard'}
            b={r['seed']:r for r in rows if r['scenario']==sc and r['policy']==pol}
            d=[a[i]['net']-b[i]['net'] for i in range(seeds)]
            se=statistics.stdev(d)/math.sqrt(seeds) if seeds>1 else 0
            paired.append(dict(scenario=sc,comparison='queue_guard minus '+pol,n=seeds,mean_difference=statistics.mean(d),ci95_low=statistics.mean(d)-1.96*se,ci95_high=statistics.mean(d)+1.96*se))
    write('full_matched_summary.csv',summary);write('full_matched_paired.csv',paired)
    assert len(rows)==len(SC)*len(POL)*seeds
    assert all(r['net']==r['net'] for r in rows)
    assert any(r['rollback'] for r in rows)
    return summary,paired
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=200);a=p.parse_args()
    s,d=run(a.output,a.seeds)
    print('FULL SIMULATION MATCHED ACTIONS',len(SC)*len(POL)*a.seeds,'rows')
    for r in d:print(r)
