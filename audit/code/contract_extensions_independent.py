#!/usr/bin/env python3
"""Check new contract calculations using coupled best responses and payoff tables.

Independent means a different mathematical implementation. It does not mean
independent researchers, independent AI tools, or external certification.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments'))
from contract_extensions import penalty_contract, reconsideration, BASE


def race(e1, e2, w1, l1, w2, l2):
    d1, d2 = w1-l1, w2-l2
    assert min(d1, d2) > 0

    def br(y, e, d):
        return 2*e*d*y/(np.sqrt(y*y+2*e*d*y)+y)

    hi = e1*d1
    x1 = brentq(lambda x: br(br(x, e2, d2), e1, d1)-x,
                hi*1e-14, hi, xtol=2e-14, rtol=2e-14)
    x2 = br(x1, e2, d2)
    return dict(x1=x1, x2=x2, V1=w1-x1/e1, V2=w2-x2/e2,
                X=x1+x2, cost=x1*x1/(2*e1)+x2*x2/(2*e2))


def independent_penalty(tau, par):
    P, B, er, ei, ee, q = (par[k] for k in ('P', 'B', 'etaR', 'etaI', 'etaE', 'q'))
    reduced = (1-tau)*B
    rr = race(er, er, B, 0., reduced, 0.)
    private = race(er, ei, P+reduced, rr['V2'], P+rr['V1'], 0.)
    enabled = race(er, ee, B, 0., B, 0.)
    joint = race(er, er, B, 0., B, 0.)
    own_release = [P+enabled['V1'], P+rr['V1'], P/2+joint['V1']]
    own_withhold = [private['V1'], (P+reduced)/3, rr['V2']]
    differences = np.subtract(own_release, own_withhold)
    g0 = (1-q)*differences[0]+q*differences[1]
    g1 = (1-q)*differences[0]+q*differences[2]
    return np.array([g0, g1, q/(2-q)/joint['X']+2*(1-q)/(2-q)/enabled['X']])


def independent_reconsideration(rho, par):
    P, B, er, ei, ee, q = (par[k] for k in ('P', 'B', 'etaR', 'etaI', 'etaE', 'q'))
    rr = race(er, er, B, rho*B, (1-rho)*B, 0.)
    joint = race(er, er, (1-rho/2)*B, rho*B/2, (1-rho/2)*B, rho*B/2)
    enabled = race(er, ee, B, rho*B, (1-rho)*B, 0.)
    private = race(er, ei, P+B, rr['V2'], P+rr['V1'], 0.)
    L, Z, U, Y, F = P+enabled['V1'], private['V1'], P+rr['V1'], P/2+joint['V1'], rr['V2']
    # Enumerate the certified second-election outcomes for each rival election.
    solo_gain = L-max(L, Z)
    gains = [(1-q)*solo_gain+q*(z*(Y-F)+(1-z)*(U-Y)) for z in [0., .37, 1.]]
    return np.array(gains)


def main():
    rng = np.random.default_rng(20261113)
    largest_penalty_error = 0.
    largest_reconsideration_error = 0.
    for _ in range(160):
        P, ei, ee = 10**rng.uniform([-2., -2.3, -.5], [0., .5, 1.])
        q = float(rng.uniform(.1, .9))
        par = dict(P=float(P), B=1., etaR=1., etaI=float(ei), etaE=float(ee), q=q)
        tau, rho = map(float, rng.uniform([.01, .01], [.9, .95]))
        a = penalty_contract(tau, par)
        observed = np.array([a['gain0'], a['gain1'], a['time']])
        expected = independent_penalty(tau, par)
        error = float(max(abs(observed-expected)/(1+abs(expected))))
        assert error < 2e-8, (par, tau, error)
        largest_penalty_error = max(largest_penalty_error, error)
        r = reconsideration(rho, par)
        expected = independent_reconsideration(rho, par)
        error = float(max(abs(expected-r['gain'])/(1+abs(expected))))
        assert error < 2e-8, (par, rho, error)
        largest_reconsideration_error = max(largest_reconsideration_error, error)
        assert (r['gain'] > 0) == (r['baseline_gain1'] > 0)
    root = brentq(lambda t: min(independent_penalty(t, BASE)[:2]), .01, .9, xtol=1e-13)
    saved = json.loads((ROOT/'results/contract_extensions.json').read_text())
    assert abs(root-saved['penalty_boundary_numeric']) < 2e-10
    assert abs(saved['reconsideration']['baseline_threshold']-
               saved['reconsideration']['revised_threshold']) < 2e-10
    revised_root = brentq(lambda rho: independent_reconsideration(rho, BASE)[0],
                         0., .95, xtol=1e-13)
    assert abs(revised_root-saved['reconsideration']['revised_threshold']) < 2e-10
    c = saved['welfare']['cases']
    assert c[0]['welfare_gain'] == 0 and c[3]['welfare_gain'] == 0
    assert 0 < c[1]['welfare_gain'] < c[2]['welfare_gain']
    joint = race(1., 1., 1., 0., 1., 0.)
    enabled = race(1., 4., 1., 0., 1., 0.)
    h = np.array([joint['X'], enabled['X']])
    costs = np.array([joint['cost'], enabled['cost']])
    weights = np.array([1/3, 2/3])
    s0 = saved['welfare']['S0']
    def direct_welfare(s, k, delta):
        return float(1+sum(weights*(k*h*s-costs*s*s)/(delta+h*s)))
    def finite_marginal(k, delta):
        step = 1e-4
        return (direct_welfare(s0-2*step, k, delta)-8*direct_welfare(s0-step, k, delta)
                +8*direct_welfare(s0+step, k, delta)-direct_welfare(s0+2*step, k, delta))/(12*step)
    k_boundary = brentq(lambda k: finite_marginal(k, .1), .1, 20., xtol=1e-8)
    d_boundary = brentq(lambda d: finite_marginal(8., d), .001, 1., xtol=1e-9)
    assert abs(k_boundary-saved['welfare']['K2_boundary_at_delta_point1']) < 2e-7
    assert abs(d_boundary-saved['welfare']['delta_boundary_at_K2_8']) < 2e-8
    for row in c:
        before = min(row['social_stake'], s0)
        after = min(row['social_stake'], saved['welfare']['Smax_numeric'])
        difference = direct_welfare(after, row['K2'], row['delta'])-direct_welfare(before, row['K2'], row['delta'])
        assert abs(difference-row['welfare_gain']) < 2e-10
    result = dict(passed=True, parameter_draws=160,
                  max_scaled_penalty_error=largest_penalty_error,
                  max_scaled_reconsideration_error=largest_reconsideration_error,
                  penalty_boundary_independent=float(root), seed=20261113,
                  reconsideration_boundary_independent=float(revised_root),
                  welfare_K2_boundary_independent=float(k_boundary),
                  welfare_delta_boundary_independent=float(d_boundary),
                  implementation='Coupled best responses; independent certified-state payoff enumeration at three rival release probabilities; no production scalar-hazard or release-gain formulas.',
                  scope='Finite numerical checks; reconsideration allows one additional election, not unrestricted disclosure timing.')
    (ROOT/'audit/output/contract_extensions_independent.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
