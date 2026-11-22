#!/usr/bin/env python3
"""Check n-team race against direct coupled best responses, not scalar G.

Finite random checks are not a proof. Reproducible seed; original first-event
payoffs are used directly and each player's loss reward is winner-independent.
"""
from pathlib import Path
import sys, math
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from n_team_race import contest
from model import contest as two_team_contest

def br(y,eta,w,l,r):
    h=eta*(r*w+y*(w-l));a=r+y
    if h==0:return 0.
    return 2*h/(math.hypot(a,math.sqrt(2*h))+a)

def direct(eta,B,rho,omega,r,outside):
    d=(1-rho)*B; ell=rho*omega*B+outside;w=ell+d
    active=np.flatnonzero(w>0)
    def residual(z):
        x=np.zeros(len(eta));x[active]=np.exp(z);total=x.sum()
        return np.array([math.log(br(total-x[i],eta[i],w[i],ell[i],r)/x[i]) for i in active])
    init=np.log(np.maximum(eta[active]*w[active]/len(eta),1e-6))
    fit=least_squares(residual,init,xtol=2e-13,ftol=2e-13,gtol=2e-13,max_nfev=500)
    assert np.linalg.norm(residual(fit.x),np.inf)<1e-8
    x=np.zeros(len(eta));x[active]=np.exp(fit.x)
    return x

def relative(a,b):return np.max(np.abs(a-b)/np.maximum(1.,np.maximum(abs(a),abs(b))))
rng=np.random.default_rng(20261103);err=0.;err2=0.;scale=0.;nchecked=0; margins=[]
for z in range(350):
    n=int(rng.integers(2,11));eta=10**rng.uniform(-1,1,n);B=10**rng.uniform(-.5,.5)
    omega=rng.dirichlet(np.ones(n));omega[rng.random(n)<.45]=0.
    if not omega.any():omega[0]=1.
    omega/=omega.sum();outside=(10**rng.uniform(-1,0,n))*(rng.random(n)<.3)
    r=0. if z%3==0 else 10**rng.uniform(-2,.5)
    rho=1. if r>0 and z%7==0 else rng.uniform(0,.98)
    a=contest(eta,B,rho,omega,r,outside);x=direct(eta,B,rho,omega,r,outside)
    err=max(err,relative(a['intensity'],x));nchecked+=1
    if n==2:
        d=(1-rho)*B;ell=rho*omega*B+outside
        t=two_team_contest(eta[0],eta[1],d+ell[0],ell[0],d+ell[1],ell[1],r)
        err2=max(err2,relative(a['intensity'],np.array([t['x1'],t['x2']])))
    if r==0:
        b=contest(eta,B,0.,omega,0.,outside)
        scale=max(scale,relative(a['intensity'],(1-rho)*b['intensity']))
    else:
        grid=np.array([0.,.25,.5,.75,.95,1.])
        h=np.array([contest(eta,B,u,omega,r,outside)['hazard'] for u in grid])
        margins.append(float(np.min((h[:-1]-h[1:])/np.maximum(h[:-1],1e-12))))
        assert np.all(np.diff(h)<0), (n,eta,B,omega,r,outside,h)
for n in range(2,11):
    eta=.7;B=1.3;rho=.37;m=max(1,n//2);omega=np.array([1/m]*m+[0.]*(n-m))
    v=contest([eta]*n,B,rho,omega)
    d=(1-rho)*B
    assert relative(v['intensity'],np.full(n,2*eta*d*(n-1)/(2*n-1)))<1e-11
    assert abs(v['value'][0]-(rho*B/m+d/(2*n-1)))<1e-11
assert max(err,err2,scale)<1e-8,(err,err2,scale)
print(f'{nchecked} n-team direct coupled best-response checks; max intensity error {err:.3g}; two-team error {err2:.3g}; undiscounted scaling error {scale:.3g}; min sampled positive-discount fractional hazard drop {min(margins):.3g}.')
