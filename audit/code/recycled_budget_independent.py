#!/usr/bin/env python3
"""Independent best-response check of recycled-prize disclosure gains and outside stakes."""
import math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from recycled_budget_analysis import calculate

def br(y,e,w,l):
 a=y;k=e*y*(w-l)
 return 2*k/(math.hypot(a,math.sqrt(2*k))+a)
def race(e1,e2,w1,l1,w2,l2):
 def f(x):return br(br(x,e2,w2,l2),e1,w1,l1)-x
 hi=max(1.,e1*(w1-l1))
 while f(hi)>0:hi*=2
 x=brentq(f,1e-15,hi,xtol=1e-15,rtol=1e-13);y=br(x,e2,w2,l2)
 return ((x*w1+y*l1-x*x/(2*e1))/(x+y),
         (y*w2+x*l2-y*y/(2*e2))/(x+y),x+y)
def stage(P,B,eR,eI,eE,rho):
 O,F,_=race(eR,eR,B,rho*B,(1-rho)*B,0)
 Q,_,_=race(eR,eR,(1-rho/2)*B,rho*B/2,(1-rho/2)*B,rho*B/2)
 E,_,_=race(eR,eE,B,rho*B,(1-rho)*B,0)
 C,_,_=race(eR,eR,P+B,0,P+B,0)
 Z,_,_=race(eR,eI,P+B,F,P+O,0)
 return dict(L=P+E,Z=Z,U=P+O,C=C,Y=P/2+Q,F=F)
rng=np.random.default_rng(20261103); err=0.;errk=0.
for _ in range(300):
 P,K,eR,eI,eE=10**rng.uniform(-1,1,5);q,rho=rng.uniform(.02,.98,2);b=rng.uniform(0,.48)*K
 par=dict(P=P,K=K,etaR=eR,etaI=eI,etaE=eE,q=q)
 v=[stage(P,K-n*b,eR,eI,eE,rho) for n in (0,1,2)]
 g0=(1-q)*(v[1]['L']+b-v[0]['Z'])+q*(v[1]['U']+b-v[0]['C'])
 g1=(1-q)*(v[1]['L']+b-v[0]['Z'])+q*(v[2]['Y']+b-v[1]['F'])
 a=calculate(par,b,rho);err=max(err,abs(g0-a['gain0']),abs(g1-a['gain1']))
 kappa=10**rng.uniform(-1,1)
 def hazard(r):return race(eR,eE,K+kappa,r*K,(1-r)*K+kappa,0)[2]
 errk=max(errk,abs(hazard(0)/hazard(1)-(K+kappa)/kappa))
assert err<1e-8,(err,errk)
assert errk<1e-8,(err,errk)
print(f'300 independent best-response parameter draws; maximum absolute gain error {err:.3g}; outside-stake ratio error {errk:.3g}.')
