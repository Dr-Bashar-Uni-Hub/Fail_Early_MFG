"""F4 synthetic decision framework: exact finite-horizon Bayes oracle and policies.
No claim of algorithmic novelty or industrial validation.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import comb
import random

@dataclass(frozen=True)
class Config:
    horizon:int=8
    batch:int=6
    alpha:int=2
    beta:int=18
    modified_alpha:int=2
    modified_beta:int=22
    quality_limit:float=.12
    risk_limit:float=.10
    pilot_cost:float=.8
    delay_cost:float=.2
    modify_cost:float=1.5
    deploy_cost:float=3.
    rollback_cost:float=4.
    benefit:float=3.
    defect_loss:float=24.
    max_pilots:int=2
    delay:int=1

class Engine:
    def __init__(self,cfg=Config()):
        self.c=cfg
        assert cfg.delay==1, 'This implementation supports one-period reporting delay'
        self._value=lru_cache(None)(self._value_uncached)
        self._actions=lru_cache(None)(self._actions_uncached)

    def mean(self,a,b): return a/(a+b)
    def pmf(self,a,b,n):
        # Beta-binomial predictive distribution
        def rising(x,k):
            v=1.
            for i in range(k):v*=x+i
            return v
        den=rising(a+b,n)
        return tuple(comb(n,k)*rising(a,k)*rising(b,n-k)/den for k in range(n+1))
    def tail(self,a,b,x):
        # integer beta shapes; beta CDF = binomial tail
        n=a+b-1
        return sum(comb(n,k)*x**k*(1-x)**(n-k) for k in range(a))
    def admissible(self,a,b):return self.tail(a,b,self.c.quality_limit)<=self.c.risk_limit
    def stage_reward(self,a,b):return self.c.benefit-self.c.defect_loss*self.mean(a,b)
    def _actions_uncached(self,t,a,b,mode,pilots,modified):
        c=self.c
        if t>=c.horizon:return ()
        if mode=='deployed':return ('continue','rollback')
        actions=['abandon','deploy']
        if pilots<c.max_pilots and t+1+c.delay<=c.horizon: actions.append('test')
        if not modified and t+1<c.horizon: actions.append('modify')
        return tuple(actions)
    def qvalues(self,t,a,b,mode='pilot',pilots=0,modified=False):
        c=self.c
        result={}
        for action in self._actions(t,a,b,mode,pilots,modified):
            if action=='abandon':
                # Tested candidate must be restored; never-tested abandoned candidate costs zero
                result[action]= -c.rollback_cost if pilots>0 else 0.
            elif action=='rollback': result[action]=-c.rollback_cost
            elif action=='modify':
                # New configuration's evidence is NOT pooled: independent prior assumption.
                result[action]=-c.modify_cost+self._value(t+1,c.modified_alpha,c.modified_beta,'pilot',0,True)
            elif action=='deploy':
                # An initial risk gate is an engineering constraint, not a theorem about safety.
                if not self.admissible(a,b):result[action]=float('-inf')
                else:result[action]=-c.deploy_cost+self._value(t,a,b,'deployed',pilots,modified)
            elif action=='test':
                # A test consumes one period, then one reporting-delay period; no deployment payoff.
                # Decision maker cannot act on its outcome before the delay ends.
                future=sum(p*self._value(t+1+c.delay,a+k,b+c.batch-k,'pilot',pilots+1,modified)
                           for k,p in enumerate(self.pmf(a,b,c.batch)))
                result[action]=-c.pilot_cost-c.delay_cost+future
            elif action=='continue':
                future=sum(p*self._value(t+1,a+k,b+c.batch-k,'deployed',pilots,modified)
                           for k,p in enumerate(self.pmf(a,b,c.batch)))
                result[action]=self.stage_reward(a,b)+future
        return result
    def _value_uncached(self,t,a,b,mode,pilots,modified):
        if t>=self.c.horizon:return 0.
        return max(self.qvalues(t,a,b,mode,pilots,modified).values())
    def decide(self,t,a,b,mode='pilot',pilots=0,modified=False):
        qs=self.qvalues(t,a,b,mode,pilots,modified)
        if not qs:return ('terminal',0.)
        action=max(qs,key=qs.get)
        return action,qs[action]
    def initial(self):return self.decide(0,self.c.alpha,self.c.beta)

def simulate(engine,seed,true_p,policy='oracle'):
    """Coupled outcome streams per seed; a modified configuration has its own true p.
    Modified true defect probability is supplied as second entry in true_p tuple.
    """
    c=engine.c
    rng=random.Random(seed)
    # Common streams by (configuration,period) regardless of policy and intervention order.
    stream={(m,t):sum(rng.random()<true_p[m] for _ in range(c.batch))
            for m in range(2) for t in range(c.horizon)}
    t=0;a=c.alpha;b=c.beta;pilots=0;modified=False;mode='pilot';net=0.;history=[]
    while t<c.horizon:
        if policy=='oracle':action,_=engine.decide(t,a,b,mode,pilots,modified)
        elif policy=='fixed':
            if mode=='deployed':action='continue'
            elif pilots<1 and t+2<=c.horizon:action='test'
            else:action='deploy' if engine.admissible(a,b) else 'abandon'
        elif policy=='risk_gate':
            if mode=='deployed':
                action='rollback' if not engine.admissible(a,b) else 'continue'
            elif pilots<2 and t+2<=c.horizon:action='test'
            else:action='deploy' if engine.admissible(a,b) else 'abandon'
        else:raise ValueError(policy)
        if action=='deploy' and not engine.admissible(a,b): action='abandon'
        history.append((t,mode,action,a,b))
        if action=='abandon':
            if pilots>0:net-=c.rollback_cost
            break
        if action=='rollback':net-=c.rollback_cost;break
        if action=='modify':
            net-=c.modify_cost;modified=True;a=c.modified_alpha;b=c.modified_beta;pilots=0;t+=1
        elif action=='test':
            net-=c.pilot_cost+c.delay_cost
            k=stream[(int(modified),t)]
            t+=1+c.delay;a+=k;b+=c.batch-k;pilots+=1
        elif action=='deploy':net-=c.deploy_cost;mode='deployed'
        elif action=='continue':
            k=stream[(int(modified),t)]
            net+=c.benefit-c.defect_loss*k/c.batch
            a+=k;b+=c.batch-k;t+=1
        else:raise RuntimeError(action)
    return {'net_value':net,'history':history,'deployed':any(h[2]=='deploy' for h in history),
            'modified':modified,'rollback':any(h[2]=='rollback' for h in history),
            'pilot_count':sum(h[2]=='test' for h in history)}
