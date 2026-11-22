#!/usr/bin/env python3
"""Cross-check the capped-budget reduction with the general race solver.

Random feasible designs do not constitute a mathematical proof or establish
that a numerical optimizer has found a global maximum.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from model import certified, welfare
from common_budget_global import analysis

rng=np.random.default_rng(20261103)
params=[dict(P=.2,cap=1.,etaR=1.,etaI=.05,etaE=4.,q=.5),
        dict(P=.01,cap=1.,etaR=1.,etaI=6.,etaE=7.,q=.83)]
checked=0
for par in params:
    result=analysis(par,2001)
    args=dict(P=par['P'],etaR=par['etaR'],etaI=par['etaI'],etaE=par['etaE'],q=par['q'],r=0.)
    base=certified(**args,B=1,rho=0)
    pi=np.array([base['pi11'],base['pi10']])
    h=np.array([base['QQ']['X'],base['EE']['X']])
    g=np.array([base['QQ']['cost'],base['EE']['cost']])
    for _ in range(600):
        b=rng.uniform(0,par['cap']/2*.999)
        B=par['cap']-2*b
        rho=rng.uniform(0,.999)
        c=certified(**args,B=B,rho=rho,bonusD=b)
        if c['gain1'] <= 0: continue
        assert c['gain0'] > 0 and c['p']==1
        S=B*(1-rho)
        assert S<=result['Smax_numeric']+1e-7
        assert np.isclose(c['t2'],np.sum(pi/h)/S,rtol=1e-10,atol=1e-10)
        for K2 in (2.,8.):
            delta=.1
            formula=1+np.sum(pi*(K2*h*S-g*S*S)/(delta+h*S))
            assert np.isclose(welfare(c,K1=1,K2=K2,delta=delta),formula,rtol=1e-10,atol=1e-10)
            row=next(x for x in result['welfare_table'] if x['K2']==K2)
            assert formula <= row['welfare_budget_design']+1e-7
        checked+=1
print(f'Passed: {checked} random strictly release-secured designs; two parameter sets; direct contest-solver time and welfare comparisons.')
