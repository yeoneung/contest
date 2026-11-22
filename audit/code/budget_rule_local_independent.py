#!/usr/bin/env python3
"""Audit local accounting-rule slopes using directly solved best responses.

No production race or gain formulas enter the direct finite differences.
"""
import math
import sys
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from budget_rule_comparison import local_comparison


def race(e1,e2,w1,l1,w2,l2):
    def br(y,e,w,l):
        c=e*y*(w-l)
        return 2*c/(math.hypot(y,math.sqrt(2*c))+y)
    f=lambda x:br(br(x,e2,w2,l2),e1,w1,l1)-x
    hi=max(1.,e1*(w1-l1))
    x=brentq(f,1e-15,hi,xtol=1e-15,rtol=1e-13)
    y=br(x,e2,w2,l2);X=x+y
    return (x*w1+y*l1-x*x/(2*e1))/X,(y*w2+x*l2-y*y/(2*e2))/X,X


def direct(par):
    P,K,eR,eI,eE,q=(par[k] for k in ('P','K','etaR','etaI','etaE','q'))
    def stage(B,rho):
        O,F,_=race(eR,eR,B,rho*B,(1-rho)*B,0.)
        E,_,XE=race(eR,eE,B,rho*B,(1-rho)*B,0.)
        Q,_,XQ=race(eR,eR,(1-rho/2)*B,rho*B/2,(1-rho/2)*B,rho*B/2)
        Z,_,_=race(eR,eI,P+B,F,P+O,0.)
        return dict(L=P+E,Y=P/2+Q,F=F,Z=Z,XE=XE,XQ=XQ)
    def gain(b,rho,recycle):
        c0=stage(K if recycle else K-2*b,rho)
        c1=stage(K-b if recycle else K-2*b,rho)
        c2=stage(K-2*b,rho)
        return (1-q)*(c1['L']+b-c0['Z'])+q*(c2['Y']+b-c1['F'])
    def time(b,recycle):
        rho=brentq(lambda z:gain(b,z,recycle),0.,.99,xtol=1e-14)
        c1=stage(K-b if recycle else K-2*b,rho);c2=stage(K-2*b,rho)
        return q/(2-q)/c2['XQ']+2*(1-q)/(2-q)/c1['XE']
    if gain(0.,0.,True)>=0 or gain(0.,.99,True)<=0:return None
    T=time(0.,True);b=K*1e-6
    slopes=[(time(b,r)-T)/b for r in (False,True)]
    return T,slopes


rng=np.random.default_rng(20261108)
params=[dict(P=.2,K=1.,etaR=1.,etaI=.05,etaE=4.,q=.5)]
for _ in range(80):
    params.append(dict(P=10**rng.uniform(-1.5,-.4),K=1.,etaR=1.,
                       etaI=10**rng.uniform(-1.7,-.2),etaE=10**rng.uniform(.1,1.),q=rng.uniform(.15,.7)))
errors=[];baseline=None
for par in params:
    d=direct(par)
    if d is None:continue
    a=local_comparison(par)
    expected=np.array([a['fixed_time_derivative'],a['recycling_time_derivative']])
    error=float(np.max(np.abs(np.array(d[1])-expected)/(1+np.abs(expected))))
    assert abs(d[0]-a['T0'])<1e-9,(par,d,a)
    assert error<5e-5,(par,error)
    errors.append(error)
    if baseline is None:baseline=dict(analytical=expected.tolist(),direct_finite_difference=d[1])
assert len(errors)>=25,len(errors)
out=dict(cases=len(errors),max_scaled_slope_error=max(errors),baseline=baseline,
         method='Direct coupled stationary best responses and separately recomputed incentive thresholds; one-sided bonus step 1e-6 of cap.')
(ROOT/'audit'/'output'/'budget_rule_local_independent.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
