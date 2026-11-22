#!/usr/bin/env python3
"""Check the added theorem and speed magnitudes with coupled best responses.

Only saved production outputs are read. No production model, slope, gain,
threshold, or optimization implementation is imported by this audit.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.optimize import brentq

from budget_region_map_independent import race, independent

ROOT = Path(__file__).resolve().parents[2]


def state(P, eI, eE, B, rho):
    O, F, _ = race(1., 1., B, rho*B, (1-rho)*B, 0.)
    E, EF, XE = race(1., eE, B, rho*B, (1-rho)*B, 0.)
    Q, QF, XQ = race(1., 1., (1-rho/2)*B, rho*B/2,
                      (1-rho/2)*B, rho*B/2)
    Z, _, _ = race(1., eI, P+B, F, P+O, 0.)
    return dict(L=P+E, Y=P/2+Q, F=F, Z=Z, XE=XE, XQ=XQ,
                costE=XE*(B-E-EF), costQ=XQ*(B-Q-QF))


def boundary_design(P, q, eI, eE, b, recycle=False):
    def stages(rho):
        if not recycle:
            s = state(P, eI, eE, 1-2*b, rho)
            return s, s, s
        return tuple(state(P, eI, eE, B, rho) for B in [1., 1-b, 1-2*b])
    def gain(rho):
        c0, c1, c2 = stages(rho)
        return (1-q)*(c1['L']+b-c0['Z']) + q*(c2['Y']+b-c1['F'])
    rho = 0. if gain(0.) >= 0 else brentq(gain, 0., 1-1e-9, xtol=1e-13)
    _, c1, c2 = stages(rho)
    piQ, piE = q/(2-q), 2*(1-q)/(2-q)
    time = piQ/c2['XQ'] + piE/c1['XE']
    w = 1 + piQ*(8*c2['XQ']-c2['costQ'])/(.1+c2['XQ'])
    w += piE*(8*c1['XE']-c1['costE'])/(.1+c1['XE'])
    return rho, time, w


def main():
    saved = json.loads((ROOT/'results/accounting_strength/checks.json').read_text())
    limit_rows = saved['limiting_theorem_checks']['rows']
    ids = np.linspace(0, len(limit_rows)-1, 40, dtype=int)
    errors = []
    signs = []
    for i in ids:
        row = limit_rows[i]
        for finite in row['finite']:
            if finite['mI'] not in (1e-4, 1e-8) or 'computed' not in finite:
                continue
            entry = dict(priority_ratio=row['priority_ratio'], q=row['q'],
                         mI=finite['mI'], mE=row['mE'], **finite['computed'])
            result = independent(entry)
            errors.append(result['scaled_slope_error'])
            if row['mE'] >= 1 and finite['mI'] == 1e-8:
                assert entry['slope_fixed'] < 0 < entry['slope_recycling']
                signs.append(True)
    with (ROOT/'results/accounting_strength/magnitude_grid.csv').open() as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    chosen = set(np.linspace(0, len(rows)-1, 36, dtype=int))
    for q in (.2, .5, .8):
        selected = [i for i, r in enumerate(rows) if r['q'] == q]
        chosen.add(max(selected, key=lambda i: rows[i]['speed_gain_percent']))
    magnitude_errors = []
    grid_checks = 0
    for i in sorted(chosen):
        row = rows[i]
        args = (.2, row['q'], row['mI'], row['mE'])
        _, t0, _ = boundary_design(*args, 0.)
        rho, t1, _ = boundary_design(*args, row['b'])
        err = max(abs(t0-row['time_attribution'])/(1+t0),
                  abs(t1-row['time_reservation'])/(1+t1), abs(rho-row['rho']))
        assert err < 2e-8, (row, err)
        assert abs(100*(1-t1/t0)-row['speed_gain_percent']) < 2e-6
        # A separate coarser feasible bonus scan checks the reported candidate.
        for bonus in np.linspace(0., .49, 17):
            _, alt, _ = boundary_design(*args, float(bonus))
            assert t1 <= alt + 2e-7*(1+alt), (row, bonus, t1, alt)
            grid_checks += 1
        magnitude_errors.append(err)
    same = saved['same_bonus_baseline']
    baseline = []
    for recycle in [False, True]:
        rho, time, w = boundary_design(.2, .5, .05, 4., same['b'], recycle)
        key = 'recycling' if recycle else 'fixed'
        assert abs(time-same[key+'_time']) < 1e-9
        if recycle:
            assert abs(w-same['recycling_welfare8']) < 1e-9
        baseline.append(dict(rule=key, rho=rho, time=time, welfare8=w))
    report = dict(passed=True, slope_cases=len(errors),
                  magnitude_grid_points=len(rows),
                  magnitude_grid_sha256=hashlib.sha256((ROOT/'results/accounting_strength/magnitude_grid.csv').read_bytes()).hexdigest(),
                  max_scaled_slope_error=max(errors),
                  rare_rediscovery_sufficient_condition_cases=len(signs),
                  magnitude_cases=len(magnitude_errors), alternative_bonus_checks=grid_checks,
                  max_scaled_magnitude_error=max(magnitude_errors), same_bonus_baseline=baseline,
                  method='Coupled stationary best responses, separately constructed release-history payoffs, fresh threshold roots, and a separate feasible-bonus grid; production outputs supply comparison targets only.',
                  scope='Finite numerical checks support the analytical proof; the coarser optimization check is not a global certificate. Independent refers to implementation, not external authors.')
    (ROOT/'audit/output/accounting_strength_independent.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
