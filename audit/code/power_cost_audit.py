#!/usr/bin/env python3
"""Independent power-cost numerical audit using direct best responses."""
import json, math
from pathlib import Path
from scipy.optimize import brentq

def br(y,eta,w,l,r,a):
    rhs=eta*(r*w+y*(w-l))
    if rhs==0:return 0.0
    def f(x):return (r+y)*x**(a-1)+(1-1/a)*x**a-rhs
    hi=1.0
    while f(hi)<0:hi*=2
    return brentq(f,0,hi,xtol=1e-15,rtol=1e-13)
def race(a,r,rho,e1=1.,e2=1.,B=1.):
    w1,l1=B,rho*B;w2,l2=(1-rho)*B,0.
    def f(x):return br(br(x,e2,w2,l2,r,a),e1,w1,l1,r,a)-x
    hi=1.
    while f(hi)>0:hi*=2
    x=brentq(f,0.,hi,xtol=1e-14,rtol=1e-13);y=br(x,e2,w2,l2,r,a)
    return x,y
out=[]
for a in (1.25,1.5,1.75,2.,2.5,3.):
 for r in (.01,.05,.5):
    xend,_=race(a,r,1.)
    criterion=(a-2)*xend+(a-1)*r
    x,y=race(a,r,1-1e-3)
    difference=x+y-xend
    if abs(criterion)>1e-8 and criterion*difference<=0:raise AssertionError((a,r,criterion,difference))
    out.append({'alpha':a,'r':r,'X_at_full':xend,'criterion':criterion,'X_near_minus_full':difference})
# Reproduce the paper's curvature counterexample without using its implementation.
base=sum(race(1.5,.05,0.));at9=sum(race(1.5,.05,.9));full=sum(race(1.5,.05,1.));assert at9<full<base
result={'cases':out,'n_cases':len(out),'published_example':{'X_0':base,'X_09_div_X_0':at9/base,'X_1_div_X_0':full/base},'limitations':'Numerical root checks only; do not prove behavior for all power exponents or global equilibrium uniqueness.'}
p=Path(__file__).resolve().parents[1]/'output/power_results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'n_cases':len(out),'published_example':result['published_example']},indent=2))
