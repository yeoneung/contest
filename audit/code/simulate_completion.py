#!/usr/bin/env python3
"""Simulate the release game event-by-event and compare to closed-form means."""
import importlib.util, json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_audit',p/'code/independent_audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(480517);N=200000
out=[]
for rho in (.0,.34,.45):
 P0,B,eR,eI,eE,q=.2,1.,1.,.05,4.,.5
 g=m.game(P0,B,eR,eI,eE,q,rho);prob=g['p']; p11=q/(2-q)
 both=rng.random(N)<p11
 d1=rng.random(N)<prob; d2=rng.random(N)<prob
 t1=np.zeros(N);t2=np.zeros(N)
 def exp(rate,n):return rng.exponential(1/rate,n)
 for k in (0,1,2):
  mask=both & ((d1.astype(int)+d2.astype(int))==k);n=mask.sum()
  rr=g['ww'] if k==0 else g['rr'] if k==1 else g['qq']
  t2[mask]=exp(rr[0]+rr[1],n)
  if k==0:t1[mask]=t2[mask]
 mask=(~both)&d1;n=mask.sum();t2[mask]=exp(g['ee'][0]+g['ee'][1],n)
 mask=(~both)&(~d1);n=mask.sum();ii=g['ii'];first=exp(ii[0]+ii[1],n);t1[mask]=first
 rediscovery=rng.random(n)<ii[1]/(ii[0]+ii[1]);second=np.zeros(n);second[rediscovery]=exp(g['rr'][0]+g['rr'][1],rediscovery.sum());t2[mask]=first+second
 p10=1-p11;X=lambda nm:g[nm][0]+g[nm][1]
 theory1=p10*(1-prob)/X('ii')+p11*(1-prob)**2/X('ww')
 theory2=p11*(prob**2/X('qq')+2*prob*(1-prob)/X('rr')+(1-prob)**2/X('ww'))+p10*(prob/X('ee')+(1-prob)*(1/X('ii')+(g['ii'][1]/X('ii'))/X('rr')))
 means=[]
 for arr,theory in ((t1,theory1),(t2,theory2)):
  se=float(arr.std(ddof=1)/np.sqrt(N));z=float((arr.mean()-theory)/se) if se else 0.
  if abs(z)>5:raise AssertionError((rho,theory,arr.mean(),z))
  means.append({'theory':theory,'simulated':float(arr.mean()),'standard_error':se,'z':z})
 out.append({'rho':rho,'p_release':prob,'method_time':means[0],'extension_time':means[1]})
(p/'output/simulation_results.json').write_text(json.dumps({'seed':480517,'n_per_share':N,'scenarios':out},indent=2)+'\n')
print(json.dumps(out,indent=2))
