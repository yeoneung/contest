#!/usr/bin/env python3
"""Exact concave bonus subproblem and numerical outer search for recycling.

The inner optimum follows a monotone first-order condition. The outer scalar
grid/refinement is numerical and is not an interval-certified global bound.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from model import certified, k_ratio
from recycled_budget_analysis import gain_fast

ROOT=Path(__file__).resolve().parents[1]


class RecyclingProfile:
    def __init__(self,par,K2=8.,delta=.1,K1=1.):
        self.par=par;self.K=par['K'];self.K2=K2;self.delta=delta;self.K1=K1
        c=certified(P=par['P'],B=1.,etaR=par['etaR'],etaI=par['etaI'],
                    etaE=par['etaE'],q=par['q'],r=0.,rho=0.)
        self.pi=np.array([c['pi11'],c['pi10']])
        self.h=np.array([c['QQ']['X'],c['EE']['X']])
        self.g=np.array([c['QQ']['cost'],c['EE']['cost']])
        ke=k_ratio(par['etaR']/par['etaE'])
        self.H=2*(1-par['q'])/(ke+2)+2*par['q']/3

    def lower(self,rho):
        return max(0.,-gain_fast(self.par,rho)/((1-rho)*self.H))

    def stakes(self,b,rho):
        return (1-rho)*(self.K-np.array([2.,1.])*b)

    def welfare(self,b,rho):
        S=self.stakes(b,rho)
        return float(self.K1+sum(self.pi*(self.K2*self.h*S-self.g*S*S)/(self.delta+self.h*S)))

    def marginal(self,b,rho):
        S=self.stakes(b,rho);d=self.delta;h=self.h;g=self.g
        vp=(self.K2*h*d-2*g*d*S-g*h*S*S)/(d+h*S)**2
        return float(sum(self.pi*np.array([2.,1.])*vp))

    def optimal_bonus(self,rho):
        lo=self.lower(rho);hi=self.K/2
        if lo>=hi:return None
        if self.marginal(lo,rho)>=0:return lo
        if self.marginal(hi,rho)<=0:return hi
        return float(brentq(lambda b:self.marginal(b,rho),lo,hi,xtol=1e-13))

    def optimum(self,points=1001):
        def objective(rho):
            b=self.optimal_bonus(rho)
            return -np.inf if b is None else self.welfare(b,rho)
        grid=np.linspace(0.,1-1e-8,points);v=np.array([objective(x) for x in grid])
        candidates=[(v[0],grid[0]),(v[-1],grid[-1])]
        for i in range(1,len(grid)-1):
            if np.isfinite(v[i]) and v[i]>=v[i-1] and v[i]>=v[i+1]:
                opt=minimize_scalar(lambda rho:-objective(rho),bounds=(grid[i-1],grid[i+1]),
                                    method='bounded',options={'xatol':1e-12})
                candidates.append((-float(opt.fun),float(opt.x)))
        value,rho=max(candidates);b=self.optimal_bonus(rho)
        return dict(K2=self.K2,rho=rho,b=b,welfare=value,grid_points=points,
                    feasible_grid_points=int(np.isfinite(v).sum()))


if __name__=='__main__':
    par=dict(P=.2,K=1.,etaR=1.,etaI=.05,etaE=4.,q=.5)
    result=dict(parameters=par,delta=.1,K1=1.,
                welfare_numeric=[RecyclingProfile(par,K2=K2).optimum() for K2 in [2.,4.,8.]],
                method='Unique concave inner bonus optimum via derivative clipping/Brent root; 1001-point outer attribution grid with local refinement. No certificate for unexamined outer intervals.')
    (ROOT/'results/recycling_profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
