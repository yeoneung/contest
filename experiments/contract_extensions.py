#!/usr/bin/env python3
"""Contract-space benchmark, one reconsideration election, and welfare sensitivity.

These are model calculations, not empirical observations. The reconsideration
game adds one election after complete certification; it is not an unrestricted
continuous-time disclosure game. Existing model algorithms are unchanged.
"""
from pathlib import Path
import json
import platform
import numpy as np
import scipy
from scipy.optimize import brentq
from model import contest, certified
from common_budget_global import analysis
from budget_rule_comparison import local_comparison

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(P=.2, B=1., etaR=1., etaI=.05, etaE=4., q=.5)


def penalty_contract(tau, parameters=None):
    p = dict(BASE if parameters is None else parameters)
    assert 0 <= tau < 1
    P, B, er, ei, ee, q = (p[k] for k in ('P', 'B', 'etaR', 'etaI', 'etaE', 'q'))
    reduced = (1-tau)*B
    ready = contest(er, er, B, 0., reduced, 0., 0.)
    private = contest(er, ei, P+reduced, ready['V2'], P+ready['V1'], 0., 0.)
    enabled = contest(er, ee, B, 0., B, 0., 0.)
    joint = contest(er, er, B, 0., B, 0., 0.)
    L, Y = P+enabled['V1'], P/2+joint['V1']
    U, F, C, Z = P+ready['V1'], ready['V2'], (P+reduced)/3, private['V1']
    g0 = (1-q)*(L-Z)+q*(U-C)
    g1 = (1-q)*(L-Z)+q*(Y-F)
    weights = np.array([q/(2-q), 2*(1-q)/(2-q)])
    hazards = np.array([joint['X'], enabled['X']])
    costs = np.array([joint['cost'], enabled['cost']])
    return dict(tau=float(tau), gain0=float(g0), gain1=float(g1), L=L, Y=Y,
                U=U, F=F, C=C, Z=Z, time=float(sum(weights/hazards)),
                welfare_K2_8_delta_point1=float(1+sum(weights*(8*hazards-costs)/(.1+hazards))))


def reconsideration(rho, parameters=None):
    par = dict(BASE if parameters is None else parameters)
    c = certified(**par, rho=rho, r=0., beta=0.)
    q, P, B = par['q'], par['P'], par['B']
    gain = q*(P+rho*B)/2+(1-q)*min(0., c['L']-c['Z'])
    p = 1. if gain > 0 else 0.
    # These times are for the strict date-zero equilibrium, away from indifference.
    t1 = c['pi10']*(1-p)/c['RI'] if c['D'] < 0 else 0.
    t2 = (c['pi11']*(p*p/c['XQ']+2*p*(1-p)/c['XR']+(1-p)**2/c['XQ'])
          +c['pi10']*(p/c['XE']+(1-p)*(1/c['RI']+(c['xI']/c['RI'])/c['XR'])))
    if c['D'] >= 0:
        t2 = c['pi11']/c['XQ']+c['pi10']/c['XE']
    return dict(rho=float(rho), gain=float(gain), date_zero_release_probability=p,
                time_method=float(t1), time_extension=float(t2), baseline_gain1=c['gain1'])


def welfare_sensitivity():
    result = analysis(dict(P=.2, cap=1., etaR=1., etaI=.05, etaE=4., q=.5))
    c = certified(**BASE)
    pi = np.array([c['pi11'], c['pi10']])
    h = np.array([c['QQ']['X'], c['EE']['X']])
    g = np.array([c['QQ']['cost'], c['EE']['cost']])
    s0, smax = result['S0'], result['Smax_numeric']

    def prime(s, k, delta):
        return float(sum(pi*(k*h*delta-2*g*delta*s-g*h*s*s)/(delta+h*s)**2))

    def welfare(s, k, delta):
        return float(1+sum(pi*(k*h*s-g*s*s)/(delta+h*s)))

    rows = []
    for k, delta in [(2., .1), (4., .1), (8., .1), (8., .01)]:
        hi = 1.
        while prime(hi, k, delta) > 0:
            hi *= 2
        social = brentq(lambda s: prime(s, k, delta), 0., hi, xtol=1e-13)
        before, after = min(social, s0), min(social, smax)
        w0, w1 = welfare(before, k, delta), welfare(after, k, delta)
        rows.append(dict(K2=k, delta=delta, social_stake=social,
                         attribution_welfare=w0, reservation_welfare=w1,
                         welfare_gain=w1-w0, welfare_gain_percent=100*(w1/w0-1)))
    return dict(S0=s0, Smax_numeric=smax,
                K2_boundary_at_delta_point1=brentq(lambda k: prime(s0, k, .1), .1, 20.),
                delta_boundary_at_K2_8=brentq(lambda d: prime(s0, 8., d), .001, 1.),
                reservation_unspent_fraction=2*result['bmax_numeric']*(1-.5)/(2-.5),
                cases=rows)


def main():
    critical = brentq(lambda t: min(penalty_contract(t)['gain0'],
                                   penalty_contract(t)['gain1']), .01, .9, xtol=1e-13)
    original = brentq(lambda z: certified(**BASE, rho=z)['gain1'], 0., .95, xtol=1e-13)
    revised = brentq(lambda z: reconsideration(z)['gain'], 0., .95, xtol=1e-13)
    equal = local_comparison(dict(P=.2, K=1., etaR=1., etaI=.05, etaE=1., q=.5))
    output = dict(parameters=BASE, penalty=penalty_contract(.4),
                  penalty_boundary_numeric=critical,
                  reconsideration=dict(baseline_threshold=original,
                                       revised_threshold=revised,
                                       zero_attribution=reconsideration(0.)),
                  welfare=welfare_sensitivity(), equal_efficiency=equal,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__),
                  scope='A fixed-cap contract benchmark and a single post-certification reconsideration election; no unrestricted-mechanism or disclosure-timing optimality claim.')
    (ROOT/'results/contract_extensions.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = [r'\begin{tabular}{rrrr}', r'\toprule',
             r'$K_2$ & $\delta$ & $S^{\mathrm{soc}}$ & Welfare gain, \%\\', r'\midrule']
    for row in output['welfare']['cases']:
        lines.append(f"${row['K2']:g}$ & ${row['delta']:g}$ & ${row['social_stake']:.4f}$ & ${row['welfare_gain_percent']:.2f}$"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'figures/data/welfare_sensitivity_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
