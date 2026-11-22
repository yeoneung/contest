#!/usr/bin/env python3
"""Fixed ex-post cap example: release bonus funded by reducing the prize.

Baseline is undiscounted; the budget covers up to two release bonuses and one
extension prize: B + 2b = K. The exogenous priority credit P is held fixed and
is outside K. Results are conditional on a unique all-release equilibrium and
compare expected completion times, not social welfare. Numerical output is an
illustration, not proof of a global optimum.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq
from model import certified

ROOT = Path(__file__).resolve().parent.parent


def calculate(par, b, margin=1e-5):
    K = par['K']; B = K - 2*b
    if not (B > 0 and b >= 0):
        raise ValueError('Require b >= 0 and B=K-2b > 0')
    args = dict(P=par['P'], B=B, etaR=par['etaR'], etaI=par['etaI'],
                etaE=par['etaE'], q=par['q'], r=0.)
    def gain(rho):
        return certified(**args, rho=rho, bonusD=b)['gain1']
    if gain(0)>0:
        threshold = 0.0
    elif gain(.999999999)>0:
        threshold = brentq(gain, 0., .999999999, xtol=1e-13)
    else:
        raise ValueError('No all-release threshold in the tested interval')
    rho = threshold+margin
    if rho >= 1:
        raise ValueError('Margin would exceed admissible share')
    z = certified(**args, rho=rho, bonusD=b)
    assert z['gain0'] > 0 and z['gain1'] > 0 and z['p'] == 1
    return dict(b=b,B=B,rho_threshold=threshold,rho=rho,
                winning_stake=(1-rho)*B,t2=z['t2'],
                release_gain0=z['gain0'],release_gain1=z['gain1'],
                max_outlay=B+2*b, outlay_when_one_ready=B+b)


def local_comparison(par):
    baseline = calculate(par,0.)
    rho=baseline['rho_threshold'];P=par['P'];K=par['K'];q=par['q']
    def d(B,rho):
        return certified(P=P,B=B,etaR=par['etaR'],etaI=par['etaI'],
                         etaE=par['etaE'],q=q,r=0.,rho=rho)['gain1']
    h=1e-5
    dr=(d(K,rho+h)-d(K,rho-h))/(2*h)
    dB=(d(K+h,rho)-d(K-h,rho))/(2*h)
    derivative=K*(1-2*dB)/dr-2*(1-rho)
    return dict(rho_threshold=rho,delta_rho=dr,delta_B=dB,
                implied_stake_derivative=derivative,
                expression='K*(1-2*delta_B)/delta_rho - 2*(1-rho_D)',
                note='Derivative at b=0 along all-release indifference; numerical central differences.')


def main():
    examples={
      'baseline':dict(P=.2,K=1.,etaR=1.,etaI=.05,etaE=4.,q=.5),
      'rediscovery_fast':dict(P=.01,K=1.,etaR=1.,etaI=6.,etaE=7.,q=.83),
    }
    results={}
    for name,par in examples.items():
        results[name]={'parameters':par,'local':local_comparison(par),
                       'attribution_only':calculate(par,0.),
                       'bonus_and_attribution':calculate(par,.02),
                       'margin_above_threshold':1e-5}
    assert results['baseline']['bonus_and_attribution']['t2'] < results['baseline']['attribution_only']['t2']
    assert results['rediscovery_fast']['bonus_and_attribution']['t2'] > results['rediscovery_fast']['attribution_only']['t2']
    output=ROOT/'results'/'fixed_budget.json';output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
