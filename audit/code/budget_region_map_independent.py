#!/usr/bin/env python3
"""Check map slopes by direct best responses and separately built payoff tables.

No map derivative, release-gain, or scalar-hazard implementation is imported.
The audit covers every grid point where both rules accelerate, a seeded sample
of the other interior points, and zero-attribution release classifications.
"""
from pathlib import Path
from collections import Counter
import csv
import json
import math
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]


def race(e1,e2,w1,l1,w2,l2):
    def br(y,e,w,l):
        c=e*y*(w-l)
        return 2*c/(math.hypot(y,math.sqrt(2*c))+y)
    f=lambda x:br(br(x,e2,w2,l2),e1,w1,l1)-x
    x=brentq(f,1e-15,max(1.,e1*(w1-l1)),xtol=1e-15,rtol=1e-13)
    y=br(x,e2,w2,l2);total=x+y
    return ((x*w1+y*l1-x*x/(2*e1))/total,
            (y*w2+x*l2-y*y/(2*e2))/total,total)


def independent(row):
    P,q,eI,eE=(float(row[k]) for k in ['priority_ratio','q','mI','mE'])
    def state(B,rho):
        O,F,_=race(1.,1.,B,rho*B,(1-rho)*B,0.)
        E,_,XE=race(1.,eE,B,rho*B,(1-rho)*B,0.)
        Q,_,XQ=race(1.,1.,(1-rho/2)*B,rho*B/2,
                     (1-rho/2)*B,rho*B/2)
        Z,_,_=race(1.,eI,P+B,F,P+O,0.)
        return dict(L=P+E,Y=P/2+Q,F=F,Z=Z,XE=XE,XQ=XQ)
    def gain(b,rho,recycle):
        c0=state(1. if recycle else 1-2*b,rho)
        c1=state(1-b if recycle else 1-2*b,rho)
        c2=state(1-2*b,rho)
        return (1-q)*(c1['L']+b-c0['Z'])+q*(c2['Y']+b-c1['F'])
    d0=gain(0.,0.,False)
    if int(row['region'])==0:
        assert d0>0,(row,d0)
        return dict(region=0,gain_error=abs(d0-float(row['gain_at_zero'])))
    rho0=brentq(lambda rho:gain(0.,rho,False),0.,1-1e-9,xtol=1e-13)
    def delay(b,recycle):
        rho=brentq(lambda z:gain(b,z,recycle),0.,1-1e-9,xtol=1e-13)
        c1=state(1-b if recycle else 1-2*b,rho)
        c2=state(1-2*b,rho)
        return q/(2-q)/c2['XQ']+2*(1-q)/(2-q)/c1['XE']
    t0=delay(0.,False)
    step=1e-4
    while max(gain(step,0.,False),gain(step,0.,True))>=0:
        step/=2
        if step<1e-10:raise AssertionError(('threshold too close to zero',row))
    slopes=[]
    for recycle in [False,True]:
        t1=delay(step,recycle);t2=delay(step/2,recycle)
        slopes.append((4*t2-t1-3*t0)/step/t0)
    expected=np.array([float(row['slope_fixed']),float(row['slope_recycling'])])
    error=float(np.max(np.abs(np.array(slopes)-expected)/(1+np.abs(expected))))
    assert abs(rho0-float(row['rhoD']))<2e-10,(row,rho0)
    assert abs(t0-float(row['T0']))/(1+t0)<1e-10,(row,t0)
    assert error<3e-5,(row,slopes,error)
    for actual,target in zip(slopes,expected):
        if abs(target)>5e-5:assert actual*target>0,(row,slopes)
    return dict(region=int(row['region']),scaled_slope_error=error)


def main():
    with (ROOT/'figures/data/budget_region_map.csv').open() as f:rows=list(csv.DictReader(f))
    rng=np.random.default_rng(20261112)
    rare=[r for r in rows if int(r['region']) in [1,3,5]]
    interior=[r for r in rows if int(r['region']) in [2,4]]
    zero=[r for r in rows if int(r['region'])==0]
    selected=rare+[interior[i] for i in rng.choice(len(interior),180,replace=False)]
    selected += [zero[i] for i in rng.choice(len(zero),36,replace=False)]
    results=[independent(row) for row in selected]
    report={'passed':True,'grid_settings':len(rows),'checked_settings':len(results),
            'checked_regions':dict(Counter(str(r['region']) for r in results)),
            'max_scaled_slope_error':max(r.get('scaled_slope_error',0) for r in results),
            'max_zero_gain_error':max(r.get('gain_error',0) for r in results),
            'method':'Direct coupled stationary best responses; separate release-history payoff tables; one-sided Richardson derivatives; seeded sample 20261112; all both-accelerate grid points included.'}
    (ROOT/'audit/output/budget_region_map_independent.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
