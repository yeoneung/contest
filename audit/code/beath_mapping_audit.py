#!/usr/bin/env python3
"""Check mapping of Beath et al. (1989) eqs. (1)--(5) to the prize race.

Illustrative and finite numerical checks; not an independent proof of novelty.
"""
import json
from math import sqrt
from pathlib import Path
import numpy as np
from scipy.optimize import brentq


def response(y,eta,w,l,r):
    a=r+y;k=eta*(r*w+y*(w-l))
    return 2*k/(sqrt(a*a+2*k)+a) if k else 0.


def equilibrium(eta,B,r,share):
    def f(x):
        y=response(x,eta,(1-share)*B,0.,r)
        return response(y,eta,B,share*B,r)-x
    hi=max(1.,eta*B,sqrt(2*eta*r*B))
    while f(hi)>0:hi*=2
    x=brentq(f,0.,hi,xtol=1e-14,rtol=1e-13)
    return x,response(x,eta,(1-share)*B,0.,r)

rng=np.random.default_rng(20261103)
max_objective_error=0.;max_reaction_error=0.;max_foc_error=0.;signs={'positive':0,'negative':0}
for _ in range(1000):
    eta,B,r=10**rng.uniform(-1,1,3)
    share=rng.uniform(.01,.98)
    x,y=equilibrium(eta,B,r,share)
    w,l=B,share*B
    # Beath's original current flow s=0 and post-innovation flows a=r*w, b=r*l.
    a_flow,b_flow=r*w,r*l
    original=((a_flow/r)*x+(b_flow/r)*y-x*x/(2*eta))/(x+y+r)
    mapped=(x*w+y*l-x*x/(2*eta))/(r+x+y)
    max_objective_error=max(max_objective_error,abs(original-mapped))
    cprime=x/eta
    foc=(w-l)*y+a_flow+x*x/(2*eta)-(x+y+r)*cprime
    max_foc_error=max(max_foc_error,abs(foc))
    analytic=(w-l-cprime)*eta/(r+x+y)
    step=1e-5*max(1.,y)
    if y<=step:step=y/2
    finite=(response(y+step,eta,w,l,r)-response(y-step,eta,w,l,r))/(2*step)
    max_reaction_error=max(max_reaction_error,abs(analytic-finite))
    signs['positive' if analytic>0 else 'negative']+=1
assert max_objective_error<1e-10 and max_foc_error<1e-9 and max_reaction_error<1e-7
critical=2-sqrt(3)
examples=[]
for share in (.2,critical,.8,.9):
    x,y=equilibrium(1.,1.,1.,share)
    examples.append({'attribution':share,'originator_intensity':x,'follower_intensity':y,
                     'originator_response_slope':(1-share-x)/(1+x+y),
                     'total_hazard':x+y})
assert examples[0]['originator_response_slope']>0 and examples[2]['originator_response_slope']<0
out={'draws':1000,'max_absolute_objective_error':max_objective_error,
     'max_absolute_foc_error':max_foc_error,'max_absolute_reaction_slope_error':max_reaction_error,
     'reaction_slope_signs':signs,'critical_share_example':critical,'examples':examples,
     'scope':['same quadratic cost for both firms to match Beath et al.; r>0 and share<1',
              'numerical checks do not establish novelty or a general proof',
              'future perpetual flow profit versus one-time prize requires s=0 and flow=discount rate times prize']}
p=Path(__file__).resolve().parents[1]/'output/beath_mapping_results.json'
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
