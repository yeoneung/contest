#!/usr/bin/env python3
"""Release bonuses under a common ex-post cap with unused funds recycled.

Designs pay b to each team that releases at date zero, and assign the remainder
K - n*b to the follow-on prize, where n is the number of such releases. Initial
readiness is as in the manuscript; later rediscoveries do not count toward n.
Results are conditional on at least one team initially being ready, with r=0.
Grid searches with numerical refinement are illustrative, not global proofs.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from model import certified, k_ratio

ROOT=Path(__file__).resolve().parent.parent

def calculate(par,b,rho):
    P,K,eR,eI,eE,q=(par[k] for k in ('P','K','etaR','etaI','etaE','q'))
    if not (0<=b<K/2 and 0<=rho<1):raise ValueError('Require b<K/2 and 0<=rho<1')
    B0,B1,B2=K,K-b,K-2*b
    args=dict(P=P,etaR=eR,etaI=eI,etaE=eE,q=q,r=0.,rho=rho)
    c0,c1,c2=[certified(**args,B=B) for B in (B0,B1,B2)]
    # Gains of a ready team's initial release versus withholding conditional
    # on the rival's initial release choice; future budgets are history-specific.
    gain0=(1-q)*(c1['L']+b-c0['Z'])+q*(c1['X']+b-c0['C'])
    gain1=(1-q)*(c1['L']+b-c0['Z'])+q*(c2['Y']+b-c1['F'])
    assert np.isclose(gain0-gain1,q*(P+rho*K)/6,atol=1e-11)
    kE=k_ratio(eR/eE)
    H=2*(1-q)/(kE+2)+2*q/3
    assert np.isclose(gain1,c0['gain1']+b*(1-rho)*H,atol=1e-11)
    # Outcome conditional on immediate release; do not use when incentives fail.
    piQ=q/(2-q);piE=2*(1-q)/(2-q)
    T=piQ/c2['QQ']['X']+piE/c1['EE']['X']
    def social(K2,delta,K1=1.):
        return K1+sum(w*(K2*z['X']-z['cost'])/(delta+z['X']) for w,z in ((piQ,c2['QQ']),(piE,c1['EE'])))
    return dict(B0=B0,B1=B1,B2=B2,gain0=gain0,gain1=gain1,H=H,
                t2=T,welfare2=social(2.,.1),welfare4=social(4.,.1),welfare8=social(8.,.1),
                max_payout=K,payout_one=B1+b,payout_two=B2+2*b)

def gain_fast(par,rho):
    P,K,eR,eI,eE,q=(par[k] for k in ('P','K','etaR','etaI','etaE','q'))
    ke=k_ratio(eR/eE)
    dR=P+(2+rho)*K/3;dI=P+(1+2*rho)*K/3
    k=k_ratio(eR*dR/(eI*dI))
    L=P+K*(rho+(1-rho)*ke/(ke+2))
    Z=P+K-2*dR/(k+2)
    return (1-q)*(L-Z)+q*(P+rho*K)/2

def search(par,samples=1001):
    P,K,eR,eI,eE,q=(par[k] for k in ('P','K','etaR','etaI','etaE','q'))
    ke=k_ratio(eR/eE);H=2*(1-q)/(ke+2)+2*q/3
    baseline=certified(P=P,B=1,etaR=eR,etaI=eI,etaE=eE,q=q,rho=0,r=0)
    pi=np.array([baseline['pi11'],baseline['pi10']])
    h=np.array([baseline['QQ']['X'],baseline['EE']['X']])
    g=np.array([baseline['QQ']['cost'],baseline['EE']['cost']])
    def lower(rho):return max(0.,-gain_fast(par,rho)/((1-rho)*H))
    def speed(rho):
        b=lower(rho)
        if b>=K/2:return np.inf
        return float((pi[0]/(h[0]*(K-2*b))+pi[1]/(h[1]*(K-b)))/(1-rho))
    def social(b,rho,K2):
        S=np.array([(K-2*b)*(1-rho),(K-b)*(1-rho)])
        return float(1+np.sum(pi*(K2*h*S-g*S*S)/(.1+h*S)))
    def optimal_welfare(rho,K2):
        lo=lower(rho);hi=K/2-1e-9
        if lo>hi:return (-np.inf,None)
        opt=minimize_scalar(lambda b:-social(b,rho,K2),bounds=(lo,hi),method='bounded')
        points=[(social(lo,rho,K2),lo),(social(hi,rho,K2),hi),(social(opt.x,rho,K2),float(opt.x))]
        return max(points)
    def refine(fun):
        grid=np.linspace(0,.999999,samples);v=np.array([fun(float(x)) for x in grid])
        cand=[(v[0],0.),(v[-1],grid[-1])]
        for i in range(1,len(grid)-1):
            if np.isfinite(v[i]) and v[i]<=v[i-1] and v[i]<=v[i+1]:
                t=minimize_scalar(fun,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12})
                cand.append((float(t.fun),float(t.x)))
        return min(cand)
    mint,rhos=refine(speed);bs=lower(rhos)
    rows=[]
    for K2 in (2.,4.,8.):
        val,rho=refine(lambda x:-optimal_welfare(x,K2)[0]);w,b=optimal_welfare(rho,K2)
        rows.append(dict(K2=K2,rho=rho,b=b,welfare=w))
    rhoD=brentq(lambda x:gain_fast(par,x),0.,.999999)
    return dict(parameters=par,grid_points=samples,H=H,attribution_only=calculate(par,0.,rhoD),
      fastest_numeric=dict(rho=rhos,b=bs,**calculate(par,bs,rhos)),
      welfare_numeric=rows,welfare_search='At every share, optimize b over the full feasible interval; strict concavity in b justifies bounded scalar optimization with endpoint comparisons.',note='Grid/refinement does not certify global optimality; equality requires all-release selection or an arbitrarily small strict margin.')

if __name__=='__main__':
    par=dict(P=.2,K=1.,etaR=1.,etaI=.05,etaE=4.,q=.5)
    result=search(par)
    (ROOT/'results'/'recycled_budget.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
