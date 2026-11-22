#!/usr/bin/env python3
"""Independent FOC-Jacobian check of the discounted total-hazard derivative."""
from pathlib import Path
from math import sqrt
import json
import numpy as np
from scipy.optimize import brentq


def br(y,e,w,l,r):
    a=r+y;k=e*(r*w+y*(w-l))
    return 2*k/(sqrt(a*a+2*k)+a) if k>0 else 0.


def race(e1,e2,w1,l1,w2,l2,r):
    def f(x):return br(br(x,e2,w2,l2,r),e1,w1,l1,r)-x
    hi=max(1.,e1*(w1-l1),sqrt(2*e1*r*w1))
    while f(hi)>0:hi*=2
    lo=1e-14 if r==0 else 0.
    x=brentq(f,lo,hi,xtol=1e-14,rtol=1e-13)
    return x,br(x,e2,w2,l2,r)


def solve(e1,e2,B,r,rho,omega,a1,a2):
    d=(1-rho)*B
    l1=rho*omega*B+a1;l2=rho*(1-omega)*B+a2
    x1,x2=race(e1,e2,d+l1,l1,d+l2,l2,r)
    R=r+x1+x2;c1=x1-e1*d;c2=x2-e2*d
    det=R*R-c1*c2
    A1=r*(1-omega)+x2;A2=r*omega+x1
    slope=-B*(e1*A1*(R-c2)+e2*A2*(R-c1))/det
    return x1+x2,slope,det,c1,c2

rng=np.random.default_rng(20261103)
maxerr=0.;positive_cross=0;min_det=float('inf')
for j in range(600):
    e1,e2,B=10**rng.uniform(-1.5,1.5,3)
    r=0. if j%5==0 else 10**rng.uniform(-3,1.5)
    rho=rng.uniform(.02,.98);omega=rng.uniform(0,1)
    a1,a2=10**rng.uniform(-2,0,2)
    z=solve(e1,e2,B,r,rho,omega,a1,a2)
    h=2e-6
    diff=(solve(e1,e2,B,r,rho+h,omega,a1,a2)[0]-solve(e1,e2,B,r,rho-h,omega,a1,a2)[0])/(2*h)
    err=abs(diff-z[1])/max(1e-8,abs(diff),abs(z[1]))
    maxerr=max(maxerr,err);positive_cross+=int(z[3]>0 or z[4]>0);min_det=min(min_det,z[2])
    if z[1]>=0 or z[2]<=0 or err>2e-5:raise AssertionError((j,z,diff,err))
out={'draws':600,'positive_jacobian_cross_term_cases':positive_cross,
     'max_relative_derivative_error':maxerr,'minimum_determinant':min_det,
     'method':'Best-response fixed point; analytic derivative from original FOC Jacobian, compared with central differences.',
     'limitations':['finite numerical checks do not prove general claims','does not audit disclosure or welfare proofs']}
p=Path(__file__).resolve().parents[1]/'output/discounted_derivative_independent.json'
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
