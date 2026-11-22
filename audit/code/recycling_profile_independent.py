#!/usr/bin/env python3
"""Check the recycling bonus envelope against direct best-response welfare."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from discounted_budget_independent import Independent

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from recycling_profile import RecyclingProfile


def main():
    rng=np.random.default_rng(20261112);cases=[];infeasible=0
    for case in range(121):
        P,eI,eE=10**rng.uniform([-2,-3,-1],[0,1,2]);q=rng.uniform(.05,.95)
        rho=rng.uniform(.01,.95);K2=10**rng.uniform(-.7,1.3);delta=10**rng.uniform(-2,.3)
        if case==120:
            P,eI,eE,q,rho,K2,delta=.2,.05,4.,.001,.8,.0001,.1
        p=dict(P=P,K=1.,etaR=1.,etaI=eI,etaE=eE,q=q)
        profile=RecyclingProfile(p,K2=K2,delta=delta)
        actual=profile.optimal_bonus(rho)
        independent=Independent(dict(priority_ratio=P,q=q,mI=eI,mE=eE,r=0.))
        hi=.5-1e-9
        def gain(b):return independent.outcome(b,rho,'R')[0]
        if gain(hi)<0:
            assert actual is None
            infeasible+=1;continue
        lo=0. if gain(0.)>=0 else brentq(gain,0.,hi,xtol=1e-13)
        assert actual is not None
        assert abs(lo-profile.lower(rho))<2e-9
        def welfare(b):
            c1=independent.state(1-b,rho);c2=independent.state(1-2*b,rho)
            return (1+q/(2-q)*(K2*c2['XQ']-c2['cQ'])/(delta+c2['XQ'])
                    +2*(1-q)/(2-q)*(K2*c1['XE']-c1['cE'])/(delta+c1['XE']))
        opt=minimize_scalar(lambda b:-welfare(b),bounds=(lo,hi),method='bounded',
                            options={'xatol':1e-12})
        expected=max(welfare(lo),welfare(hi),-float(opt.fun))
        actual_value=profile.welfare(actual,rho)
        error=abs(actual_value-expected)/(1+abs(expected))
        assert error<2e-7,(p,rho,K2,delta,actual,lo,opt,error)
        kind='lower' if abs(actual-lo)<1e-8 else 'upper_limit' if actual>.5-1e-8 else 'interior'
        cases.append(dict(kind=kind,scaled_welfare_error=error))
    output=dict(passed=True,parameter_draws=121,feasible_cases=len(cases),
                infeasible_cases=infeasible,optimizer_types=dict(Counter(c['kind'] for c in cases)),
                max_scaled_welfare_error=max(c['scaled_welfare_error'] for c in cases),
                method='Independent coupled best responses, expected-payoff welfare, independently solved release-bonus lower bound, bounded scalar optimization plus endpoint comparisons.',seed=20261112)
    (ROOT/'audit/output/recycling_profile_independent.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
