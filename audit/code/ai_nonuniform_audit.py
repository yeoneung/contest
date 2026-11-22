#!/usr/bin/env python3
"""Run finite tests of the nonuniform-efficiency comparative statics."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2]/'experiments'/'ai_nonuniform.py'),run_name='__main__')

# Positive discounting can change the direction of the release-gain response.
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from model import certified
par=dict(P=.01,B=1.,etaR=.2,etaI=.1,etaE=1.,q=.1,rho=.1,r=1.)
a=certified(**par);par['etaE']=1.1;b=certified(**par)
assert b['gain1']>a['gain1']>0
print('Positive-discount counterexample: release gains',a['gain1'],b['gain1'])

# For r>0, independently compare the implicit interior-equilibrium derivative
# to finite changes in all three technologies. The formula is local and applies
# to a selected simple interior branch, including either sign of the slope.
import numpy as np
rng=np.random.default_rng(20261104);h=1e-5;count=0;worst=0.
for _ in range(10000):
 par=dict(P=10**rng.uniform(-2,-.3),B=1.,etaR=10**rng.uniform(-.5,.5),etaI=10**rng.uniform(-1.5,.3),etaE=10**rng.uniform(-.1,1.),q=rng.uniform(.05,.7),rho=rng.uniform(.1,.6),r=10**rng.uniform(-2,0))
 try:base=certified(**par)
 except (AssertionError,ValueError):continue
 p=base['p'];slope=base['gain1']-base['gain0']
 if not (.05<p<.95 and abs(slope)>1e-5):continue
 g=rng.normal(size=3);keys=('etaR','etaE','etaI');pv=[];dv=[]
 for sign in (-1,1):
  v=par.copy()
  for key,rate in zip(keys,g):v[key]*=np.exp(sign*h*rate)
  c=certified(**v);pv.append(c['p']);dv.append(c['gain0']+(c['gain1']-c['gain0'])*p)
 predicted=-(dv[1]-dv[0])/(2*h*slope);actual=(pv[1]-pv[0])/(2*h)
 worst=max(worst,abs(actual-predicted));count+=1
 if count>=150:break
assert count>=100 and worst<1e-6,(count,worst)
print(count,'positive-discount interior-branch checks, max derivative absolute error',worst)
