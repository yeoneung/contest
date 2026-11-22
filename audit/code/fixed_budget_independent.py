#!/usr/bin/env python3
"""Independent best-response check of the equal-cap comparison.

Uses the direct quadratic expected-payoff best responses and their composition,
not code/model.py's G equation. Finite numerical checks cannot prove a theorem.
"""
from pathlib import Path
import json, math, sys
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from model import certified


def br(y,eta,w,los):
    if w==los==0:return 0.
    if y==0:return eta*(w-los) if w>los else 0.
    k=eta*y*(w-los)
    return 2*k/(math.sqrt(y*y+2*k)+y) if k>0 else 0.


def race(e1,e2,w1,l1,w2,l2):
    def f(x):return br(br(x,e2,w2,l2),e1,w1,l1)-x
    hi=max(1.,e1*(w1-l1))
    while f(hi)>0:hi*=2
    x=brentq(f,1e-15,hi,xtol=1e-14)
    y=br(x,e2,w2,l2)
    X=x+y
    v1=(x*w1+y*l1-x*x/(2*e1))/X
    v2=(y*w2+x*l2-y*y/(2*e2))/X
    return x,y,v1,v2


def direct(P,B,eR,eI,eE,q,rho,b):
    rr=race(eR,eR,B,rho*B,(1-rho)*B,0)
    qq=race(eR,eR,(1-rho/2)*B,rho*B/2,(1-rho/2)*B,rho*B/2)
    ee=race(eR,eE,B,rho*B,(1-rho)*B,0)
    ww=race(eR,eR,P+B,0,P+B,0)
    ii=race(eR,eI,P+B,rr[3],P+rr[2],0)
    L=P+ee[2]+b; Z=ii[2]; Y=P/2+qq[2]+b
    gain=(1-q)*(L-Z)+q*(Y-rr[3])
    pi11=q/(2-q);pi10=2*(1-q)/(2-q)
    t2=pi11/(qq[0]+qq[1])+pi10/(ee[0]+ee[1])
    return gain,t2


rng=np.random.default_rng(20261103)
max_gain=0.;max_t=0.;checked=0
for _ in range(200):
    P,B,eR,eI,eE=10**rng.uniform(-1.5,1.5,5);q=rng.uniform(.05,.95)
    rho=rng.uniform(.01,.95);b=rng.uniform(0,.15)*B
    gain,t=direct(P,B,eR,eI,eE,q,rho,b)
    try:
        m=certified(P=P,B=B,etaR=eR,etaI=eI,etaE=eE,q=q,rho=rho,bonusD=b)
    except ValueError:
        print('MODEL_FAILURE:',dict(P=P,B=B,eR=eR,eI=eI,eE=eE,q=q,rho=rho,b=b)); raise
    max_gain=max(max_gain,abs(gain-m['gain1']))
    t_ref=m['pi11']/m['XQ']+m['pi10']/m['XE']
    max_t=max(max_t,abs(t-t_ref)/max(1,t_ref))
    checked+=1

scenarios=[]
for P,eI,eE,q in [(.2,.05,4.,.5),(.01,6.,7.,.83)]:
    K=eR=1.
    def root(b):
        B=K-2*b
        f=lambda rho: direct(P,B,eR,eI,eE,q,rho,b)[0]
        return brentq(f,0,.9999999,xtol=1e-13)
    rho0=root(0.);h=1e-5
    def delta(B,rho):return direct(P,B,eR,eI,eE,q,rho,0)[0]
    dr=(delta(K,rho0+h)-delta(K,rho0-h))/(2*h)
    db=(delta(K+h,rho0)-delta(K-h,rho0))/(2*h)
    formula=K*(1-2*db)/dr-2*(1-rho0)
    stake=lambda b:(1-root(b))*(K-2*b)
    numerical=(stake(h)-stake(0))/h
    _,t0=direct(P,K,eR,eI,eE,q,rho0+1e-5,0)
    rb=root(.02)+1e-5
    _,t1=direct(P,.96,eR,eI,eE,q,rb,.02)
    scenarios.append(dict(rho0=rho0,formula=formula,forward_difference=numerical,
                          t2_attribution=t0,t2_bonus=t1,all_release_share=rb))
assert max_gain < 1e-9 and max_t < 1e-9
assert all(abs(s['formula']-s['forward_difference']) < 1e-4 for s in scenarios)
assert scenarios[0]['formula'] > 0 > scenarios[1]['formula']
res=dict(draws=checked,max_abs_gain_error=max_gain,max_relative_time_error=max_t,
         examples=scenarios,limitations=['finite numerical verification is not proof',
         'focuses on discounted rate zero and positive prizes',
         'does not prove global optimality or evaluate welfare'])
out=ROOT/'audit/output/fixed_budget_independent.json'
out.write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
