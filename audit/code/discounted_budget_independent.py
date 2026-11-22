#!/usr/bin/env python3
"""Discounted-accounting checks from direct coupled best responses.

No production contest solver, continuation table, or slope is imported.
"""
from pathlib import Path
from functools import lru_cache
from collections import Counter
import csv
import json
import math
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]


def race(e1,e2,w1,l1,w2,l2,r):
    def br(y,e,w,l):
        c=e*(r*w+y*(w-l));a=r+y
        return 2*c/(math.sqrt(a*a+2*c)+a) if c>0 else 0.
    f=lambda x:br(br(x,e2,w2,l2),e1,w1,l1)-x
    lo=0. if r>0 else 1e-15
    x=brentq(f,lo,max(1.,e1*w1),xtol=1e-15,rtol=1e-13)
    y=br(x,e2,w2,l2);X=x+y
    cost=x*x/(2*e1)+y*y/(2*e2)
    v1=(x*w1+y*l1-x*x/(2*e1))/(r+X)
    v2=(y*w2+x*l2-y*y/(2*e2))/(r+X)
    return v1,v2,X,cost


class Independent:
    def __init__(self,row,public=False):
        self.P=float(row.get('priority_ratio',.2));self.q=float(row.get('q',.5))
        self.eI=float(row.get('mI',.05));self.eE=float(row.get('mE',4.))
        self.r=float(row['r']);self.public=public
        self.state=lru_cache(maxsize=512)(self._state)

    def _state(self,B,rho):
        P,r=self.P,self.r
        O,F,_,_=race(1.,1.,B,rho*B,(1-rho)*B,0.,r)
        E,_,XE,cE=race(1.,self.eE,B,rho*B,(1-rho)*B,0.,r)
        Q,_,XQ,cQ=race(1.,1.,(1-rho/2)*B,rho*B/2,(1-rho/2)*B,rho*B/2,r)
        C,_,_,_=race(1.,1.,P+B,0.,P+B,0.,r)
        Z,_,_,_=race(1.,self.eI,P+B,F,P+O,0.,r)
        return dict(L=P+E,U=P+O,Y=P/2+Q,C=C,Z=Z,F=F,XE=XE,XQ=XQ,cE=cE,cQ=cQ)

    def outcome(self,b,rho,rule):
        c=[self.state(1-n*b if rule=='R' else 1-2*b,rho) for n in [0,1,2]]
        solo=c[1]['L']+b-c[0]['Z'];rw=c[1]['U']+b-c[0]['C'];rd=c[2]['Y']+b-c[1]['F']
        gains=[solo,rw,rd] if self.public else [(1-self.q)*solo+self.q*rw,(1-self.q)*solo+self.q*rd]
        T=self.q/(2-self.q)/c[2]['XQ']+2*(1-self.q)/(2-self.q)/c[1]['XE']
        return min(gains),T

    def boundary_time(self,b,rule):
        top=1. if self.r>0 else 1-1e-8
        if self.outcome(b,0.,rule)[0]>=0:
            return None
        rho=brentq(lambda z:self.outcome(b,z,rule)[0],0.,top,xtol=1e-13,rtol=1e-13)
        return rho,self.outcome(b,rho,rule)[1]

    def check(self,row):
        region=int(row['region']);gain0,_=self.outcome(0.,0.,'F')
        if region==0:
            assert gain0>0
            return dict(region=region,gain_zero_error=abs(gain0-float(row['gain_at_zero'])))
        rho,T0=self.boundary_time(0.,'F')
        assert abs(rho-float(row['rhoD']))<2e-9
        assert abs(T0-float(row['T0']))/(1+T0)<2e-9
        step=1e-4
        while any(self.boundary_time(step,j) is None for j in ['F','R']):
            step/=2
            assert step>1e-10
        slopes=[]
        for j in ['F','R']:
            t1=self.boundary_time(step,j)[1];t2=self.boundary_time(step/2,j)[1]
            slopes.append((4*t2-t1-3*T0)/step/T0)
        expected=np.array([float(row['slope_fixed']),float(row['slope_recycling'])])
        error=float(np.max(abs(np.array(slopes)-expected)/(1+abs(expected))))
        assert error<2e-5,(row,slopes,error)
        for x,y in zip(slopes,expected):
            if abs(y)>2e-5:assert x*y>0
        return dict(region=region,scaled_slope_error=error)


def main():
    with (ROOT/'figures/data/discounted_budget_map.csv').open() as f:rows=list(csv.DictReader(f))
    rng=np.random.default_rng(20261112)
    # Include all rare sign patterns, then seeded samples of common patterns and gray cases.
    selected=[r for r in rows if int(r['region']) in [1,3,5]]
    for region,n in [(2,80),(4,50),(0,30)]:
        pool=[r for r in rows if int(r['region'])==region]
        selected.extend(pool[i] for i in rng.choice(len(pool),min(n,len(pool)),replace=False))
    assert not any(int(r['region'])==5 for r in selected),'Boundary points require a separate directional audit.'
    checked=[Independent(row).check(row) for row in selected]
    report=json.loads((ROOT/'results/discounted_budget_robustness.json').read_text())
    timing=[]
    for row in report['baseline_and_timing']:
        timing.append(Independent(row,public=row['certification_before_election']).check(row))
    output=dict(passed=True,grid_settings=len(rows),checked_grid_settings=len(checked),
                checked_regions=dict(Counter(str(x['region']) for x in checked)),
                checked_timing_cases=len(timing),
                max_scaled_slope_error=max(x.get('scaled_slope_error',0) for x in checked+timing),
                method='Direct coupled best responses and expected payoffs; history-specific prizes; independently resolved incentive boundaries and right-sided Richardson derivatives; no production solver or slope imports.',
                seed=20261112)
    (ROOT/'audit/output/discounted_budget_independent.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
