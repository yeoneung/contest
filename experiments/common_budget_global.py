#!/usr/bin/env python3
"""Numerical illustration of the capped-budget design theorem; not data.

The continuous scalar search is numerical, not a proof of a global optimum.
The theorem concerns the supremum over designs that induce immediate release.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from model import certified, k_ratio

ROOT = Path(__file__).resolve().parent.parent


def analysis(par, samples=1001):
    P, cap, eR, eI, eE, q = (par[k] for k in ('P','cap','etaR','etaI','etaE','q'))
    kE = k_ratio(eR/eE)
    def gain(B, rho):
        k = k_ratio(eR*(P+(2+rho)*B/3)/(eI*(P+(1+2*rho)*B/3)))
        dR = P+(2+rho)*B/3
        L = P+B*(rho+(1-rho)*kE/(kE+2))
        Z = P+B-2*dR/(k+2)
        return (1-q)*(L-Z)+q*(P+rho*B)/2
    def threshold(b):
        B = cap-2*b
        if gain(B,0)+b >= 0: return 0.
        return brentq(lambda rho: gain(B,rho)+b,0,1,xtol=1e-13)
    def stake(b): return (cap-2*b)*(1-threshold(b))
    initial = certified(P=P,B=cap,etaR=eR,etaI=eI,etaE=eE,q=q,rho=0,r=0)
    weights = np.array([initial['pi11'], initial['pi10']])
    h = np.array([initial['QQ']['X'], initial['EE']['X']])/cap
    g = np.array([initial['QQ']['cost'], initial['EE']['cost']])/cap**2
    rho0 = threshold(0.); S0 = stake(0.)
    grid = np.linspace(0, cap/2, samples)
    values = np.array([stake(b) if b < cap/2 else 0. for b in grid])
    # Refine every grid-detected local maximum; compare against endpoints.
    candidates = [(values[0],0.)]
    for i in range(1,len(grid)-1):
        if values[i]>=values[i-1] and values[i]>=values[i+1]:
            res=minimize_scalar(lambda b:-stake(b),bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-14})
            candidates.append((-res.fun,float(res.x)))
    Smax,bmax=max(candidates)
    def times(S): return float(np.sum(weights/h)/S)
    def welfare(S,K2,delta): return float(1+np.sum(weights*(K2*h*S-g*S*S)/(delta+h*S)))
    def derivative(S,K2,delta): return float(np.sum(weights*(K2*h*delta-2*g*delta*S-g*h*S*S)/(delta+h*S)**2))
    table=[]
    for K2 in (1.,2.,4.,8.):
        delta=.1
        hi=1.
        while derivative(hi,K2,delta)>0: hi*=2
        Sfb=brentq(lambda x:derivative(x,K2,delta),0,hi)
        before=min(Sfb,S0); after=min(Sfb,Smax)
        table.append(dict(K2=K2,delta=delta,social_stake=Sfb,attribution_stake=before,
            budget_design_stake=after,welfare_attribution=welfare(before,K2,delta),
            welfare_budget_design=welfare(after,K2,delta),marginal_at_S0=derivative(S0,K2,delta)))
    # Independently check analytical release gain and proportionality against the general contest solver.
    for b,rho in ((0.,.12),(.02,.34),(.15,.03)):
        B=cap-2*b
        c=certified(P=P,B=B,etaR=eR,etaI=eI,etaE=eE,q=q,rho=rho,r=0,bonusD=b)
        assert abs(gain(B,rho)+b-c['gain1'])<1e-10
        S=B*(1-rho)
        assert np.allclose([c['QQ']['X'],c['EE']['X']],h*S,rtol=1e-11)
        assert np.allclose([c['QQ']['cost'],c['EE']['cost']],g*S*S,rtol=1e-11)
    return dict(parameters=par, grid_points=samples,rho0=rho0,S0=S0,bmax_numeric=bmax,
                rho_at_bmax=threshold(bmax),Smax_numeric=Smax,
                expected_time_at_S0=times(S0),expected_time_at_Smax=times(Smax),
                welfare_table=table, note='Numerical scan/refinement is not a certificate of the global maximum; threshold designs require equilibrium selection or an arbitrarily small strict-release margin.')

if __name__=='__main__':
    result=analysis(dict(P=.2,cap=1.,etaR=1.,etaI=.05,etaE=4.,q=.5))
    out=ROOT/'results'/'common_budget_global.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
