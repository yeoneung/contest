#!/usr/bin/env python3
"""Exact expectation of finite piecewise-constant effort deviations."""
import importlib.util,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('direct_solver',P/'code/independent_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(221031);maxgain=-float('inf');nonlocalviolations=0;count=0
for j in range(1000):
 eta1,eta2=10**rng.uniform(-2,2,2);d1,d2=10**rng.uniform(-2,2,2);l1,l2=10**rng.uniform(-2,2,2);r=0. if j%4==0 else 10**rng.uniform(-2,2)
 x,y,v,_=m.race(eta1,eta2,d1+l1,l1,d2+l2,l2,r)
 for k in range(4):
  rates=x*10**rng.uniform(-2,2,4);durations=10**rng.uniform(-2,2,3)/(r+x+y)
  val=0.;survival=1.
  for effort,dt in zip(rates[:3],durations):
   h=r+effort+y;flow=effort*(d1+l1)+y*l1-effort**2/(2*eta1)
   z=h*dt;val+=survival*flow/h*(-math.expm1(-z));survival*=math.exp(-z)
  effort=rates[-1];h=r+effort+y;flow=effort*(d1+l1)+y*l1-effort**2/(2*eta1)
  val+=survival*flow/h
  gain=val-v;maxgain=max(maxgain,gain);count+=1
  if gain>1e-8*max(1,abs(v)):nonlocalviolations+=1
result={'seed':221031,'races':1000,'piecewise_deviations':count,'maximum_deviation_gain':maxgain,'violations':nonlocalviolations,'limitations':'Tests only a finite selection of deterministic piecewise-constant deviations; does not verify every admissible process or nonstationary equilibrium.'}
(P/'output/deviations.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if nonlocalviolations:raise SystemExit(1)
