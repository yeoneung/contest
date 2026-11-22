#!/usr/bin/env python3
"""Independent numerical checks using best responses, not the paper's scalar root.
Numerical checks are not proofs. The deterministic seed makes runs reproducible.
"""
from pathlib import Path
import json, math
import numpy as np
from scipy.optimize import brentq
P=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(20261101)

def br(y,eta,w,l,r):
    # Direct maximization of (x*w+y*l-x²/(2eta))/(r+x+y).
    a=r+y; k=eta*(r*w+y*(w-l))
    if k==0: return 0.0
    return 2*k/(math.hypot(a,math.sqrt(2*k))+a)

def race(e1,e2,w1,l1,w2,l2,r):
    def f(x): return br(br(x,e2,w2,l2,r),e1,w1,l1,r)-x
    if w1==0: x=0.0
    else:
        lo=1e-15 if r==0 else 0.0
        hi=max(1.0,e1*(w1-l1),math.sqrt(2*e1*r*w1))
        while f(hi)>0:hi*=2
        x=brentq(f,lo,hi,xtol=1e-15,rtol=1e-13,maxiter=200)
    y=br(x,e2,w2,l2,r)
    X=x+y
    if X+r<=0: raise ValueError('No event and no discount; outside nondegenerate domain')
    V1=(x*w1+y*l1-x*x/(2*e1))/(r+X)
    V2=(y*w2+x*l2-y*y/(2*e2))/(r+X)
    return x,y,V1,V2

def paramrace(e1,e2,B,rho,r,omega,a1=0,a2=0):
    d=(1-rho)*B
    return race(e1,e2,d+rho*omega*B+a1,rho*omega*B+a1,d+rho*(1-omega)*B+a2,rho*(1-omega)*B+a2,r)

def game(P0,B,eR,eI,eE,q,rho,r=0,beta=0):
    b=beta*B
    rr=race(eR,eR,B+b,rho*B+b,(1-rho)*B,0,r)
    qq=race(eR,eR,(1-rho/2)*B+b/2,rho*B/2+b/2,(1-rho/2)*B+b/2,rho*B/2+b/2,r)
    ee=race(eR,eE,B+b,rho*B+b,(1-rho)*B,0,r)
    ww=race(eR,eR,P0+B+b,0,P0+B+b,0,r)
    ii=race(eR,eI,P0+B+b,rr[3],P0+rr[2],0,r)
    L=P0+ee[2]; Z=ii[2]; U=P0+rr[2]; F=rr[3]; Y=P0/2+qq[2]; C=ww[2]
    d0=(1-q)*(L-Z)+q*(U-C); d1=(1-q)*(L-Z)+q*(Y-F)
    p=1. if d0>=0 and d1>=0 else 0. if d0<=0 and d1<=0 else d0/(d0-d1)
    return {'d0':d0,'d1':d1,'p':p,'rr':rr,'qq':qq,'ee':ee,'ww':ww,'ii':ii,'L':L,'Z':Z}

def geom():return 10**rng.uniform(-2,2)
def rel(a,b):return abs(a-b)/max(abs(a),abs(b),1e-13)

fails=[]; errors={'best_response':0.,'zero_discount_scaling':0.,'timescale':0.,'continuation_zero_discount':0.,'threshold_endpoint':0.,'completion_bonus_invariance':0.,'full_share_limit':0.};counts={}
# Two direct best responses and direct expected-value checks over arbitrary outcomes.
for j in range(1500):
    e1,e2=geom(),geom();d1,d2=geom(),geom();l1,l2=geom(),geom();r=0. if j%4==0 else geom()
    w1,w2=l1+d1,l2+d2
    x,y,v1,v2=race(e1,e2,w1,l1,w2,l2,r)
    err=max(rel(x,br(y,e1,w1,l1,r)),rel(y,br(x,e2,w2,l2,r)))
    errors['best_response']=max(errors['best_response'],err)
    # A direct nonlocal grid of unilateral constant deviations, independent of root formula.
    for xi,other,eta,w,l,v in [(x,y,e1,w1,l1,v1),(y,x,e2,w2,l2,v2)]:
        choices=xi*np.array([0.01,0.1,0.5,0.9,1.,1.1,2.,10.,100.])
        vv=(choices*w+other*l-choices**2/(2*eta))/(r+choices+other)
        if np.max(vv)>v+1e-8*max(1,abs(v)):fails.append(['profitable_deviation',j])
counts['arbitrary_payoff_races']=1500
# Attribution monotonicity and zero-discount scaling, including corners.
for j in range(650):
    e1,e2,B=geom(),geom(),geom();r=0. if j%4==0 else geom();om=rng.uniform(0,1);a1,a2=geom(),geom()
    rs=[0.,.2,.5,.8,.98,.999] + ([1.] if r>0 else [])
    sols=[paramrace(e1,e2,B,u,r,om,a1,a2) for u in rs]
    xs=np.array([v[0]+v[1] for v in sols])
    if np.any(np.diff(xs)>1e-9*max(xs)):fails.append(['quadratic_monotonicity',j,rs,xs.tolist()])
    if r==0:
        errors['zero_discount_scaling']=max(errors['zero_discount_scaling'],max(rel(xs[k],xs[0]*(1-rs[k])) for k in range(len(rs))))
    s=geom(); ref=paramrace(e1,e2,B,.37,r/s,om,a1,a2); scaled=paramrace(s*e1,s*e2,B,.37,r,om,a1,a2)
    errors['timescale']=max(errors['timescale'],rel(scaled[0]+scaled[1],s*(ref[0]+ref[1])),rel(scaled[2],ref[2]),rel(scaled[3],ref[3]))
    if r>0:
        # Sole-originator full-attribution, no outside bonuses.
        z=paramrace(e1,e2,B,1,r,1.,0,0); expected=2*e1*r*B/(math.sqrt(r*r+2*e1*r*B)+r)
        errors['full_share_limit']=max(errors['full_share_limit'],rel(z[0]+z[1],expected))
counts['attribution_parameter_draws']=650
# Timing and outcome conditioning of the originator's additional reward.
for j in range(400):
    e1,e2,B=geom(),geom(),geom(); r=0. if j%4==0 else geom(); rho=rng.uniform(0,.9)
    beta=10**rng.uniform(-2,1); gamma=rng.uniform(.1,.9)*(1-rho)
    base=paramrace(e1,e2,B,rho,r,1.)
    complete=paramrace(e1,e2,B,rho,r,1.,beta*B,0.)
    xb=base[0]+base[1]; xc=complete[0]+complete[1]
    if r==0 and rel(xb,xc)>1e-7:fails.append(['completion_bonus_changed_r0_hazard',j])
    if r>0 and xc < xb-1e-9*max(1,xb):fails.append(['completion_bonus_decreased_hazard',j])
    loss=race(e1,e2,B,(rho+gamma)*B,(1-rho)*B,0,r)
    if loss[0]+loss[1] > xb+1e-9*max(1,xb):fails.append(['loss_bonus_increased_hazard',j])
counts['payment_timing_draws']=400
# Release decisions and continuation values in undiscounted certified game.
for j in range(500):
    P0,B,eR,eI,eE=geom(),geom(),geom(),geom(),geom();q=rng.uniform(.01,.99)
    rs=[0.,.2,.4,.6,.8,.99]; g=[game(P0,B,eR,eI,eE,q,u) for u in rs]
    if np.any(np.diff([v['d0'] for v in g]) < -1e-9) or np.any(np.diff([v['d1'] for v in g]) < -1e-9): fails.append(['release_gain_nonmonotone',j])
    if np.any(np.diff([v['p'] for v in g]) < -1e-9): fails.append(['release_probability_nonmonotone',j])
    for k,u in enumerate(rs):
        z=g[k];expectedO=B*(1+2*u)/3;expectedF=B*(1-u)/3;expectedQ=B*(2+u)/6
        errors['continuation_zero_discount']=max(errors['continuation_zero_discount'],rel(z['rr'][2],expectedO),rel(z['rr'][3],expectedF),rel(z['qq'][2],expectedQ))
    if g[0]['d1']<=0 and g[-1]['d1']>0:
        root=brentq(lambda u:game(P0,B,eR,eI,eE,q,u)['d1'],0,.99)
        errors['threshold_endpoint']=max(errors['threshold_endpoint'],abs(game(P0,B,eR,eI,eE,q,root)['d1']))
    else:root=None
    u=rng.uniform(0,.98);beta=geom();A=game(P0,B,eR,eI,eE,q,u,0,beta);C=game(P0+beta*B,B,eR,eI,eE,q,u,0,0)
    for nm in ['d0','d1']:
        errors['completion_bonus_invariance']=max(errors['completion_bonus_invariance'],rel(A[nm],C[nm]))
    for nm in ['rr','qq','ee','ww','ii']:
        for k in [0,1]:errors['completion_bonus_invariance']=max(errors['completion_bonus_invariance'],rel(A[nm][k],C[nm][k]))
counts['release_game_parameter_draws']=500
for name,value in errors.items():
    if value > 1e-7: fails.append(['identity_tolerance_exceeded',name,value])
result={'seed':20261101,'methods':'Quadratic best responses derived directly from expected payoffs; coupled equilibrium solved by composition; independent of manuscript scalar root and code/model.py.', 'counts':counts,'max_relative_errors_except_threshold_absolute':errors,'n_flagged_failures':len(fails),'first_flagged_failures':fails[:5],'limitations':['finite-precision numerical tests do not establish a theorem','tests use stationary constant intensities; arbitrary-process deviations are not independently verified','not a comprehensive review of all theorems or of nonstationary equilibria','random draws restricted to nondegenerate model assumptions']}
(P/'output/results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if fails:raise SystemExit(1)
