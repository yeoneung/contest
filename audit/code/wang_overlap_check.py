#!/usr/bin/env python3
"""Reproduce payoff mapping and a high-attribution example in Wang comparison.

Our two-player post-release race has B=1, r=1, c_i(x)=x²/2.
This numerical example is not a proof or empirical evidence. Wang imposes an
upper effort bound and further cost/stability assumptions, so the quadratic
calculation only illustrates the issue; it is not an asserted direct instance
of all of Wang's assumptions.
"""
import json
import math
from pathlib import Path
from scipy.optimize import brentq


def best_response(y, winning, losing, discount=1.0):
    a = discount + y
    k = discount * winning + y * (winning - losing)
    return 2 * k / (math.sqrt(a * a + 2 * k) + a) if k else 0.0


def race(share):
    def f(x):
        y = best_response(x, 1 - share, 0)
        return best_response(y, 1, share) - x
    x = brentq(f, 0, 2, xtol=1e-14)
    y = best_response(x, 1 - share, 0)
    return {'share': share, 'originator_intensity': x,
            'follower_intensity': y, 'total_hazard': x + y,
            'originator_loss_reward': share}


solo = 2 - math.sqrt(3)  # max_x (x - x²/2)/(1+x)
rows = [race(s) for s in (0, .2, .3, .8, .9)]
assert rows[-1]['originator_intensity'] > rows[-2]['originator_intensity']
assert all(a['total_hazard'] > b['total_hazard'] for a, b in zip(rows, rows[1:]))
result = {'B': 1, 'r': 1, 'eta_R': 1, 'eta_E': 1,
          'originator_standalone_value': solo, 'results': rows,
          'interpretation': 'At high attribution, originator effort can rise while total hazard falls. The originator loss prize exceeds the standalone value, outside the stated complementarity condition of Wang Corollary 6.',
          'limitations': ['illustrative computation, not a proof', 'unbounded quadratic effort is not literally Wang Assumption 10', 'the standalone threshold is for these parameters only']}
out = Path(__file__).resolve().parents[1] / 'output/wang_overlap_check.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
