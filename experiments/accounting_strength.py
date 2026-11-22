#!/usr/bin/env python3
"""Reproduce the accounting theorem checks and speed-gain sensitivity study.

Run from any directory. Outputs are in results/accounting_strength/. All results
are model calculations, not data estimates. The original numerical modules
are imported without modification.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/accounting_strength"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments"))
from model import k_ratio
from budget_region_map import local_effects
from common_budget_global import analysis
from recycled_budget_analysis import calculate, gain_fast


def limiting_case(priority, q, mE):
    """Normalize cap=eta_R=1. These are limits, not an eta_I=0 game."""
    k = k_ratio(1 / mE)
    kappa = k / (k + 2)
    A = (1 - q) * (1 - kappa)
    lam = A + q / 2
    rho = (A - q * priority / 2) / lam
    if rho <= 0:
        return None
    hQ = 4 / 3
    hE = 2 * (k + 1) / (k * (k + 2))
    piQ, piE = q / (2 - q), 2 * (1 - q) / (2 - q)
    Q, E = piQ / hQ, piE / hE
    H = A + 2 * q / 3
    sf = -(1 - q) / (lam * (1 - rho))
    sr = (2 * Q + E) / (Q + E) - H / lam
    condition = 9 * hE * (A + q / 3) - 4 * (1 - q)
    assert sf < 0
    assert np.sign(sr) == np.sign(condition)
    if mE >= 1:
        assert condition > 0
    return dict(rhoD=rho, slope_fixed=sf, slope_recycling=sr,
                D_rho=lam, D_B=-q*priority/2,
                recycling_sign_margin=condition, kE=k, A=A)


def verify_limit():
    rows = []
    skipped = 0
    levels = [1e-4, 1e-8, 1e-12, 1e-16]
    errors = {str(e): [] for e in levels}
    for priority in [.05, .2, .5]:
        for q in [.1, .2, .5, .8, .9]:
            for mE in np.geomspace(.1, 100, 17):
                lim = limiting_case(priority, q, float(mE))
                if lim is None:
                    skipped += 1
                    continue
                row = dict(priority_ratio=priority, q=q, mE=float(mE), limit=lim)
                row["finite"] = []
                for mI in levels:
                    exact = local_effects(priority, q, mI, float(mE))
                    if exact["region"] in (0, 5):
                        row["finite"].append(dict(mI=mI, already_release=True))
                        continue
                    discrepancies = {key: abs(exact[key] - lim[key]) / (1 + abs(lim[key]))
                                     for key in ["rhoD", "slope_fixed", "slope_recycling", "D_rho", "D_B"]}
                    errors[str(mI)].append(max(discrepancies.values()))
                    row["finite"].append(dict(mI=mI, normalized_max_error=max(discrepancies.values()),
                                              computed=exact))
                    if mI == levels[-1]:
                        assert exact["slope_fixed"] < 0
                        assert np.sign(exact["slope_recycling"]) == np.sign(lim["slope_recycling"])
                        assert max(discrepancies.values()) < 1e-6
                rows.append(row)
    return dict(cases=len(rows), noninterior_limit_cases_excluded=skipped,
                rediscovery_ratios=levels,
                max_scaled_errors={e: max(v) for e, v in errors.items()},
                rows=rows)


def magnitude_grid(points_i=81, points_e=81):
    grid_i = np.geomspace(.001, 10, points_i)
    grid_e = np.geomspace(.1, 100, points_e)
    rows = []
    summaries = []
    for q in [.2, .5, .8]:
        panel = []
        for j, mE in enumerate(grid_e):
            for mI in grid_i:
                par = dict(P=.2, cap=1., etaR=1., etaI=float(mI), etaE=float(mE), q=q)
                local = local_effects(.2, q, float(mI), float(mE))
                if local['gain_at_zero'] >= 0:
                    # With release at rho=0, stake=cap is already maximal:
                    # every feasible (cap-2b)*(1-rho) is at most cap.
                    ke = k_ratio(1 / float(mE))
                    hE = 2*(ke+1)/(ke*(ke+2))
                    t0 = q/(2-q)/(4/3) + 2*(1-q)/(2-q)/hE
                    out = dict(expected_time_at_Smax=t0, expected_time_at_S0=t0,
                               rho0=0., bmax_numeric=0., rho_at_bmax=0.)
                else:
                    out = analysis(par, samples=1001)
                gain = 100 * (1 - out["expected_time_at_Smax"] / out["expected_time_at_S0"])
                assert gain > -1e-7
                panel.append(dict(q=q, mI=float(mI), mE=float(mE),
                                  rhoD=out["rho0"], b=out["bmax_numeric"],
                                  rho=out["rho_at_bmax"], speed_gain_percent=max(0., gain),
                                  time_attribution=out["expected_time_at_S0"],
                                  time_reservation=out["expected_time_at_Smax"]))
            if (j+1) % 10 == 0:
                print(f"q={q:g}: {j+1}/{points_e} method-use rows", flush=True)
        def summary(data):
            gains = [x["speed_gain_percent"] for x in data]
            return dict(cases=len(data), median=float(np.median(gains)),
                        p90=float(np.quantile(gains, .9)), max=float(np.max(gains)),
                        cases_positive_gain=sum(x > 1e-5 for x in gains))
        summaries.append(dict(q=q, all_grid=summary(panel),
                              positive_threshold_only=summary([x for x in panel if x["rhoD"] > 1e-10])))
        rows.extend(panel)
        print(f"Magnitude grid q={q:g}: {len(panel)} cases", flush=True)
    with (OUT / "magnitude_grid.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(priority_ratio=.2, readiness=[.2, .5, .8],
                rediscovery_ratio=dict(min=.001, max=10, count=points_i, spacing="geometric"),
                method_use_ratio=dict(min=.1, max=100, count=points_e, spacing="geometric"),
                objective="Minimize expected extension time among immediate-release fixed-reservation rules; not welfare maximization",
                search="1001-point bonus scan with local refinement; numerical optimization, not a global certificate",
                denominator="Attribution-only time at the minimum immediate-release share, or zero share where release already holds",
                interpretation="Quantiles weight grid cells equally; they are not population frequencies or empirical effect estimates",
                summaries=summaries)


def same_bonus():
    par = dict(P=.2, cap=1., etaR=1., etaI=.05, etaE=4., q=.5)
    fixed = analysis(par, samples=1001)
    recycling = dict(par)
    recycling["K"] = recycling.pop("cap")
    b = fixed["bmax_numeric"]
    k = k_ratio(par["etaR"] / par["etaE"])
    H = 2 * (1 - par["q"]) / (k + 2) + 2 * par["q"] / 3
    rho = brentq(lambda x: gain_fast(recycling, x) + b * (1-x) * H, 0., .999999, xtol=1e-13)
    rec = calculate(recycling, b, rho)
    assert abs(rec["gain1"]) < 1e-10
    return dict(b=b, attribution_only_time=fixed["expected_time_at_S0"],
                reservation_stake_bound=fixed['Smax_numeric'],
                social_stake=fixed['welfare_table'][-1]['social_stake'],
                reservation_welfare=fixed['welfare_table'][-1]['welfare_budget_design'],
                reservation_expected_spending=1-2*b*(1-par['q'])/(2-par['q']),
                fixed_rho=fixed["rho_at_bmax"], fixed_time=fixed["expected_time_at_Smax"],
                recycling_rho=rho, recycling_time=float(rec["t2"]), recycling_welfare8=float(rec["welfare8"]),
                recycling_slower_than_reservation_percent=100*(rec["t2"]/fixed["expected_time_at_Smax"]-1),
                recycling_slower_than_attribution_only_percent=100*(rec["t2"]/fixed["expected_time_at_S0"]-1),
                selection="Both rules use their own minimum release-supporting share at the same bonus; boundary outcomes are approached with strict release margins")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points-i', type=int, default=81)
    parser.add_argument('--points-e', type=int, default=81)
    args = parser.parse_args()
    if min(args.points_i, args.points_e) < 3:
        parser.error('Both grid dimensions must be at least 3.')
    result = dict(date="2026-11-21", kind="Model calculations; no empirical data",
                  limiting_theorem_checks=verify_limit(), magnitude_grid=magnitude_grid(args.points_i,args.points_e),
                  same_bonus_baseline=same_bonus())
    (OUT / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = {k: v for k, v in result.items() if k != "limiting_theorem_checks"}
    compact["limiting_theorem_checks"] = {k: v for k, v in result["limiting_theorem_checks"].items() if k != "rows"}
    print(json.dumps(compact, indent=2))
