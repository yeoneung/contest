#!/usr/bin/env python3
"""Compare public-release and readiness-conditioned winning awards.

Certification continues to inform play. The public-history award ignores the
readiness certificate in its payout. All quantities are model calculations.
"""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.optimize import brentq
from model import contest
from budget_region_map import local_effects
from contract_extensions import penalty_contract

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(P=.2, cap=1., q=.5, etaR=1., etaI=.05, etaE=4., tau=.4)


def calculate(par):
    P, cap, q, er, ei, ee, tau = (par[k] for k in
                                 ['P','cap','q','etaR','etaI','etaE','tau'])
    assert 0 <= tau < 1
    low = (1-tau)*cap
    E = contest(er, ee, cap, 0., low, 0.)
    O = contest(er, er, cap, 0., low, 0.)
    Q = contest(er, er, cap, 0., cap, 0.)
    A = contest(er, er, low, 0., low, 0.)
    I = contest(er, ei, P+low, A['V2'], P+A['V1'], 0.)
    L, Z, C, Y = P+E['V1'], I['V1'], (P+low)/3, P/2+Q['V1']
    g0 = (1-q)*(L-Z)+q*(P+O['V1']-C)
    g1 = (1-q)*(L-Z)+q*(Y-O['V2'])
    piQ, piE = q/(2-q), 2*(1-q)/(2-q)
    time = piQ/Q['X'] + piE/E['X']
    welfare = 1 + sum(p*(8*z['X']-z['cost'])/(.1+z['X'])
                      for p,z in [(piQ,Q),(piE,E)])
    spending = piQ*cap + piE*(E['x1']*cap+E['x2']*low)/E['X']
    return dict(parameters=par, gain0=float(g0), gain1=float(g1),
                unique_release=bool(min(g0,g1)>0),
                time_if_release=float(time), welfare_if_release=float(welfare),
                expected_spending_if_release=float(spending))


def main():
    baseline = calculate(BASE)
    assert baseline['unique_release']
    rows=[]
    for q, ei, ee, tau in itertools.product([.2,.5,.8],[.01,.05,.5],[.5,1.,4.],[.2,.4,.6]):
        rows.append(calculate(dict(BASE,q=q,etaI=ei,etaE=ee,tau=tau)))
    boundary = brentq(lambda x:local_effects(.2,.5,x,4.)['slope_fixed'],.01,.31,xtol=1e-13)
    bracket=[]
    for mI in [boundary-.001,boundary+.001]:
        bracket.append(dict(priority_ratio=.2,q=.5,mI=mI,mE=4.,
                            **local_effects(.2,.5,mI,4.)))
    assert bracket[0]['slope_fixed']<0<bracket[1]['slope_fixed']
    certified = penalty_contract(.4)
    result = dict(date='2026-11-21',
        public_rule='Winner i receives cap*(1-tau+tau*D_i), where D_i indicates date-zero release.',
        certification_for_play_retained=True,
        readiness_used_in_public_payout=False,
        baseline=baseline, comparison_cases=rows,
        certified_baseline=dict(time=certified['time'],
             welfare=certified['welfare_K2_8_delta_point1'],expected_spending=1.),
        baseline_rediscovery_boundary=dict(priority_ratio=.2,q=.5,method_use_ratio=4.,
             ratio_at_zero_reservation_slope=boundary,bracket=bracket),
        interpretation='Winning stakes may differ under a contract based only on public research history. The two award extensions are feasible examples, not optimal mechanisms over a broader contract space. Outcomes labeled if_release are equilibrium outcomes only when both gains are positive.')
    (ROOT/'results/public_history_awards.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(baseline=baseline,comparison_cases=len(rows),
                         baseline_boundary=boundary),indent=2))


if __name__ == '__main__':
    main()
