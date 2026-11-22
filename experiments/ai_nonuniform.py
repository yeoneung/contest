#!/usr/bin/env python3
"""Analytical comparison of nonuniform efficiencies in the 2-team r=0 model.

Log derivatives and finite comparisons are exact only within the stipulated
model. The quantities are not empirical AI estimates.
"""
from pathlib import Path
import sys, math
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from model import k_ratio,certified

def elasticity(k):return 2+k/(k+2)-2*k/(2*k+1)
def components(P,B,rho,mE,mI,q):
    dR=P+(2+rho)*B/3;dI=P+(1+2*rho)*B/3
    kE=k_ratio(1/mE);kI=k_ratio(dR/(mI*dI))
    N=-2*B*(1-rho)/(kE+2)+2*dR/(kI+2)
    A_E=(1-q)*2*B*(1-rho)*kE/((kE+2)**2*elasticity(kE))
    A_I=(1-q)*2*dR*kI/((kI+2)**2*elasticity(kI))
    gamma=q*(P+rho*B)/6
    return dict(N=N,A_E=A_E,A_I=A_I,ratio=A_E/A_I,kE=kE,kI=kI,gamma=gamma)
def gap(P,B,rho,mE,mI,q,p):
    c=components(P,B,rho,mE,mI,q)
    return (1-q)*c['N']+q*(P+rho*B)*(2/3-p/6)
def threshold(P,B,mE,mI,q):
    f=lambda rho:gap(P,B,rho,mE,mI,q,1.)
    return 0. if f(0)>=0 else brentq(f,0,1-1e-10)

if __name__=='__main__':
    rng=np.random.default_rng(20261104);worst=0.;worst_finite=0.;worst_tau=0.
    for _ in range(1000):
        P,B=10**rng.uniform(-2,1,2);rho=rng.uniform(.02,.96);mE,mI=10**rng.uniform(-2,2,2);q=rng.uniform(.05,.95);p=rng.uniform(0,1)
        c=components(P,B,rho,mE,mI,q);h=1e-5
        dE=(gap(P,B,rho,mE*math.exp(h),mI,q,p)-gap(P,B,rho,mE*math.exp(-h),mI,q,p))/(2*h)
        dI=(gap(P,B,rho,mE,mI*math.exp(h),q,p)-gap(P,B,rho,mE,mI*math.exp(-h),q,p))/(2*h)
        worst=max(worst,abs(dE+c['A_E']),abs(dI-c['A_I']))
        facE,facI=10**rng.uniform(-.4,.4,2)
        actual=certified(P=P,B=B,etaR=1,etaI=mI*facI,etaE=mE*facE,q=q,rho=rho)
        old=certified(P=P,B=B,etaR=1,etaI=mI,etaE=mE,q=q,rho=rho)
        analytic=(1-q)*(components(P,B,rho,mE*facE,mI*facI,q)['N']-c['N'])
        worst_finite=max(worst_finite,abs((actual['gain1']-old['gain1'])-analytic),abs((actual['gain0']-old['gain0'])-analytic))
        if _<100:
            D=threshold(P,B,mE,mI,q)
            if D>1e-7:
                F=gap(P,B,D,mE*facE,mI*facI,q,1.)
                D2=threshold(P,B,mE*facE,mI*facI,q)
                if F>1e-8: assert D2<D+1e-9
                if F< -1e-8: assert D2>D-1e-9
    assert worst<1e-7 and worst_finite<1e-9,(worst,worst_finite)
    print('1000 random draws: max local derivative absolute error',worst,'max finite gap error',worst_finite)
    print('baseline at rho=.34:',components(.2,1,.34,4,.05,.5))
    for mE,mI in [(4,.05),(4.4,.05),(4,.055),(8,.1),(16,.1),(8,.2)]:
        c=certified(P=.2,B=1,rho=.34,q=.5,etaR=1,etaE=mE,etaI=mI)
        print('etaE',mE,'etaI',mI,'symmetric release',c['p'])

    args=dict(P=.01,B=1.,etaR=.2,etaI=.1,q=.1,rho=.1,r=1.)
    discounted=[certified(**args,etaE=e)['gain1'] for e in (1.,1.1)]
    assert discounted[1]>discounted[0]>0
    print('discounted method-use counterexample, etaE 1 to 1.1:',discounted)
