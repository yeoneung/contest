#!/usr/bin/env python3
"""Reproduce the baseline, original sensitivities, figure data and consistency checks.

Run from any directory:  python experiments/reproduce.py
Outputs
  figures/data/*.csv, figures/data/*.png   inputs for the pgfplots figures
  results/numbers.json                     every number quoted in the text
  results/verification.json                residuals of the consistency checks
All parameters are illustrative. They are not estimates.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model import contest, certified, welfare, h, k_ratio

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'figures' / 'data'
RES = ROOT / 'results'
DATA.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

BASE = dict(P=0.2, B=1.0, etaR=1.0, etaI=0.05, etaE=4.0, q=0.5)
SEED = 20261028

# sequential blue ramp (light to dark) used for every heat map
BLUES = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf',
         '#184f95', '#0d366b']
CMAP = LinearSegmentedColormap.from_list('seqblue', BLUES)


def r4(x):
    return float(np.round(x, 4))


def write_csv(name, header, rows):
    with open(DATA / name, 'w') as f:
        f.write(' '.join(header) + '\n')
        for row in rows:
            f.write(' '.join(f'{v:.6f}' if np.isfinite(v) else 'nan'
                             for v in row) + '\n')


# ----------------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------------
def unique_release(c):
    return c['gain0'] > 0 and c['gain1'] > 0


def rho_D(r=0.0, beta=0.0, **par):
    """Smallest share above which all release is the unique equilibrium."""
    p = {**BASE, **par}
    top = 0.999999 if r == 0 else 1.0
    grid = np.linspace(0, top, 2001)
    ok = np.array([unique_release(certified(**p, rho=x, r=r, beta=beta))
                   for x in grid])
    if ok.all():
        return 0.0
    i = np.where(~ok)[0][-1]
    if i == len(grid) - 1:
        return np.nan
    a, b = grid[i], grid[i + 1]

    def f(x):
        c = certified(**p, rho=x, r=r, beta=beta)
        return min(c['gain0'], c['gain1'])
    return brentq(f, a, b, xtol=1e-13)


def rho_U(r=0.0, **par):
    p = {**BASE, **par}
    f = lambda x: certified(**p, rho=x, r=r)['D']
    if f(0) >= 0:
        return 0.0
    return brentq(f, 0, 0.999999, xtol=1e-13)


def beta_D(rho=0.0, r=0.0, **par):
    p = {**BASE, **par}
    f = lambda b: min(certified(**p, rho=rho, r=r, beta=b)['gain0'],
                      certified(**p, rho=rho, r=r, beta=b)['gain1'])
    if f(0) > 0:
        return 0.0
    hi = 1.0
    while f(hi) <= 0:
        hi *= 2
    return brentq(f, 0, hi, xtol=1e-13)


def frozen_t2(c, c0):
    """Expected extension time with release from c and intensities from c0."""
    p, pi11, pi10 = c['p'], c['pi11'], c['pi10']
    RI = c0['RI']
    return (pi11 * (p * p / c0['XQ'] + 2 * p * (1 - p) / c0['XR']
                    + (1 - p) ** 2 / c0['XW'])
            + pi10 * (p / c0['XE'] + (1 - p) * (1 / RI + (c0['xI'] / RI)
                                                / c0['XR'])))


# ----------------------------------------------------------------------------
# Fixed-intensity benchmark of Appendix B
# ----------------------------------------------------------------------------
def fixed_game(P=.2, B=1., nu=1., lam=.1, mu=5., r=.1, c=.05, q=.85, rho=0.):
    H = nu * B / (r + 2 * nu)
    X = P + (1 + rho) * H
    F = (1 - rho) * H
    Y = P / 2 + H
    C = (nu * (P + B) - c) / (r + 2 * nu)
    L = P + B * (nu + rho * mu) / (r + nu + mu)
    Z = (nu * (P + B) + lam * F - c) / (r + nu + lam)
    d = (1 - q) * (L - Z) + q * (X - C)
    s = q * (r * P + 2 * c) / (2 * (r + 2 * nu))
    if s > 1e-14:
        p = float(np.clip(d / s, 0, 1))
    else:
        p = 1. if d > 1e-12 else (0. if d < -1e-12 else np.nan)
    return dict(H=H, X=X, F=F, Y=Y, C=C, L=L, Z=Z, d=d, s=s, p=p)


# ----------------------------------------------------------------------------
# Consistency checks
# ----------------------------------------------------------------------------
def checks():
    rng = np.random.default_rng(SEED)
    err = {k: 0.0 for k in [
        'contest_hjb', 'contest_foc', 'contest_closed_form_r0',
        'contest_deviation_gain', 'scaling_equivalence',
        'full_attribution_hazard', 'self_limiting_derivative',
        'certified_closed_form_r0', 'fixed_game_affine', 'monte_carlo_T']}
    flags = {k: 0 for k in [
        'G_not_decreasing', 'hazard_not_decreasing_in_rho',
        'release_fails_at_full_attribution', 'gain_not_increasing_r0',
        'rhoU_below_rhoD', 'additive_hazard_r0_changed',
        'additive_hazard_not_increasing', 'additive_gain_not_increasing_r0',
        'welfare_outside_bracket', 'comparative_statics_sign']}
    n = 400
    for _ in range(n):
        e1, e2 = np.exp(rng.uniform(-2, 2, 2))
        w1, w2 = np.exp(rng.uniform(-1.5, 1.5, 2))
        l1, l2 = w1 * rng.uniform(0, .95), w2 * rng.uniform(0, .95)
        r = 0.0 if rng.uniform() < .3 else float(np.exp(rng.uniform(-5, 1)))
        z = contest(e1, e2, w1, l1, w2, l2, r)
        for xi, xj, eta, w, l, V in [(z['x1'], z['x2'], e1, w1, l1, z['V1']),
                                     (z['x2'], z['x1'], e2, w2, l2, z['V2'])]:
            hjb = xi * (w - V) + xj * (l - V) - xi * xi / (2 * eta) - r * V
            err['contest_hjb'] = max(err['contest_hjb'], abs(hjb))
            err['contest_foc'] = max(err['contest_foc'], abs(xi - eta * (w - V)))
            # deviation to a deterministic time-varying intensity
            a, b = rng.uniform(0, 3 * xi + .5, 2)
            kappa = rng.uniform(.1, 3)
            u = lambda t: b + (a - b) * np.exp(-kappa * t)
            U = lambda t: b * t + (a - b) * (1 - np.exp(-kappa * t)) / kappa
            integrand = lambda t: np.exp(-(r + xj) * t - U(t)) * (
                u(t) * w + xj * l - u(t) ** 2 / (2 * eta))
            J = quad(integrand, 0, np.inf, limit=400)[0]
            err['contest_deviation_gain'] = max(err['contest_deviation_gain'],
                                                J - V)
        # monotonicity of G on a grid
        d1, d2 = w1 - l1, w2 - l2
        c1 = e1**2 * d1**2 - 2 * e1 * r * l1
        c2 = e2**2 * d2**2 - 2 * e2 * r * l2
        lo = max(r, np.sqrt(max(0, -c1)), np.sqrt(max(0, -c2)), 1e-6)
        Rg = np.linspace(lo * 1.0001, lo + 10 * (r + e1 * d1 + e2 * d2) + 1, 400)
        G = (r + Rg + e1 * d1 + e2 * d2 - np.sqrt(Rg**2 + c1)
             - np.sqrt(Rg**2 + c2)) / Rg
        if np.any(np.diff(G) >= 0):
            flags['G_not_decreasing'] += 1
        if r == 0:
            k = k_ratio(e1 * d1 / (e2 * d2))
            x1 = 2 * e1 * d1 / (k + 2)
            x2 = 2 * k * e2 * d2 / (2 * k + 1)
            err['contest_closed_form_r0'] = max(err['contest_closed_form_r0'],
                                                abs(x1 - z['x1']), abs(x2 - z['x2']))
        # attribution with split weights and a completion bonus
        B = 1.0
        om = rng.choice([1.0, 0.5])
        a1, a2 = (rng.uniform(0, .5), 0.0) if om == 1.0 else (rng.uniform(0, .5),) * 2
        rr = float(np.exp(rng.uniform(-5, 0)))
        rhos = np.linspace(0, 1, 41)
        Xs = [contest(e1, e2, (1 - x) * B + x * om * B + a1, x * om * B + a1,
                      (1 - x) * B + x * (1 - om) * B + a2,
                      x * (1 - om) * B + a2, rr)['X'] for x in rhos]
        if np.any(np.diff(Xs) >= 0):
            flags['hazard_not_decreasing_in_rho'] += 1
        # completion bonus: r = 0 unchanged, r > 0 increasing
        base0 = contest(e1, e2, B, 0.3 * B, .7 * B, 0, 0)['X']
        add0 = contest(e1, e2, B + .4, 0.3 * B + .4, .7 * B, 0, 0)['X']
        err_add = abs(base0 - add0)
        if err_add > 1e-9:
            flags['additive_hazard_r0_changed'] += 1
        Xa = [contest(e1, e2, B + a, .3 * B + a, .7 * B, 0, rr)['X']
              for a in np.linspace(0, 1, 11)]
        if np.any(np.diff(Xa) <= 0):
            flags['additive_hazard_not_increasing'] += 1
        # full attribution hazard formula and self limiting derivative
        z1 = contest(e1, e2, B, B, 0.0, 0.0, rr)
        err['full_attribution_hazard'] = max(
            err['full_attribution_hazard'],
            abs(z1['X'] - (np.sqrt(rr * rr + 2 * e1 * rr * B) - rr)))
        eps = 1e-6
        zm = contest(e1, e2, B, (1 - eps) * B, eps * B, 0.0, rr)
        zm2 = contest(e1, e2, B, (1 - 2 * eps) * B, 2 * eps * B, 0.0, rr)
        fd = (3 * z1['V1'] - 4 * zm['V1'] + zm2['V1']) / (2 * eps)
        Xbar = z1['X']
        formula = -(e2 / e1) * B * Xbar / (Xbar + rr)
        err['self_limiting_derivative'] = max(
            err['self_limiting_derivative'],
            abs(fd - formula) / (1 + abs(formula)))
    # certified game
    for _ in range(150):
        par = dict(P=float(np.exp(rng.uniform(-2, .5))), B=1.0, etaR=1.0,
                   etaI=float(np.exp(rng.uniform(-4, 0))),
                   etaE=float(np.exp(rng.uniform(-1, 2.5))),
                   q=float(rng.uniform(.05, .95)))
        s = float(np.exp(rng.uniform(-1.5, 1.5)))
        rr = float(np.exp(rng.uniform(-4, 0)))
        rho = float(rng.uniform(0, .95))
        c1 = certified(**par, rho=rho, r=rr)
        sp = dict(par, etaR=s * par['etaR'], etaI=s * par['etaI'],
                  etaE=s * par['etaE'])
        c2 = certified(**sp, rho=rho, r=s * rr)
        err['scaling_equivalence'] = max(
            err['scaling_equivalence'], abs(c1['gain0'] - c2['gain0']),
            abs(c1['gain1'] - c2['gain1']), abs(c1['t2'] / s - c2['t2']),
            abs(c1['t1'] / s - c2['t1']))
        cf = certified(**par, rho=1.0, r=rr)
        if not (cf['gain0'] > 0 and cf['gain1'] > 0):
            flags['release_fails_at_full_attribution'] += 1
        # r = 0 closed forms
        c0 = certified(**par, rho=rho, r=0.0)
        P, B, q = par['P'], par['B'], par['q']
        kE = k_ratio(par['etaR'] / par['etaE'])
        L = P + B * (rho + (1 - rho) * kE / (kE + 2))
        dR, dI = P + (2 + rho) * B / 3, P + (1 + 2 * rho) * B / 3
        kp = k_ratio(par['etaR'] * dR / (par['etaI'] * dI))
        Zc = P + B - 2 * dR / (kp + 2)
        A = (1 - q) * (L - Zc) + 2 * q * (P + rho * B) / 3
        Gam = q * (P + rho * B) / 6
        err['certified_closed_form_r0'] = max(
            err['certified_closed_form_r0'], abs(L - c0['L']), abs(Zc - c0['Z']),
            abs(A - c0['gain0']), abs(Gam - c0['Gamma']))
        rr_grid = np.linspace(0, .99, 34)
        g1 = [certified(**par, rho=x)['gain1'] for x in rr_grid]
        if np.any(np.diff(g1) <= 0):
            flags['gain_not_increasing_r0'] += 1
        rd, ru = rho_D_fast(par), rho_U(**par)
        if ru < rd - 1e-9 or (ru > 0 and ru <= rd):
            flags['rhoU_below_rhoD'] += 1
        # technology comparative statics at r = 0
        up = certified(**dict(par, etaE=par['etaE'] * 1.01), rho=rho)
        ui = certified(**dict(par, etaI=par['etaI'] * 1.01), rho=rho)
        if not (up['gain1'] < c0['gain1'] and ui['gain1'] > c0['gain1']):
            flags['comparative_statics_sign'] += 1
        # additive recognition gains at r = 0
        bs = np.linspace(0, 2, 9)
        cb = [certified(**par, rho=rho, beta=b) for b in bs]
        if (np.any(np.diff([c['gain0'] for c in cb]) <= 0)
                or np.any(np.diff([c['gain1'] for c in cb]) <= 0)):
            flags['additive_gain_not_increasing_r0'] += 1
        # welfare bracket in the release region
        if rd < .9:
            K2, delta = float(rng.uniform(.5, 4)), float(rng.uniform(.02, .5))
            zs = []
            for key in ['QQ', 'EE']:
                Rs, Cs = c0_ref(par)[key]
                zs.append((np.sqrt(delta**2 + K2 * delta * Rs**2 / Cs) - delta) / Rs)
            fun = lambda x: -welfare(certified(**par, rho=x), K1=1.0, K2=K2,
                                     delta=delta)
            lo, hi = rd + 1e-6, .995
            opt = minimize_scalar(fun, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-9})
            zopt = 1 - opt.x
            zlo = max(min(zs), 1 - hi)
            zhi = min(max(zs), 1 - lo)
            if zlo <= zhi and not (zlo - 1e-5 <= zopt <= zhi + 1e-5):
                flags['welfare_outside_bracket'] += 1
    # Appendix B fixed-intensity game
    for _ in range(300):
        P, B, nu, lam, mu = np.exp(rng.uniform(-2, 2, 5))
        r, c = rng.uniform(0, .4, 2)
        q, rho, p = rng.uniform(.01, .99), rng.uniform(0, 1), rng.uniform()
        g = fixed_game(P, B, nu, lam, mu, r, c, q, rho)
        gain = (1 - q) * (g['L'] - g['Z']) + q * ((1 - p) * (g['X'] - g['C'])
                                                 + p * (g['Y'] - g['F']))
        err['fixed_game_affine'] = max(err['fixed_game_affine'],
                                       abs(gain - (g['d'] - g['s'] * p)))
    # Monte Carlo check of the publication and extension time formulas
    err['monte_carlo_T'] = monte_carlo(rng)
    # Proposition 7: completion bonus equals a priority increase when r = 0,
    # Proposition 8: the release bonus shifts the gain one for one,
    # and higher priority credit raises the value of withholding
    err['completion_priority_equivalence'] = 0.0
    err['release_bonus_shift'] = 0.0
    flags['dZdP_nonpositive'] = 0
    for _ in range(150):
        par = dict(P=float(np.exp(rng.uniform(-2, .5))), B=1.0, etaR=1.0,
                   etaI=float(np.exp(rng.uniform(-4, 0))),
                   etaE=float(np.exp(rng.uniform(-1, 2.5))),
                   q=float(rng.uniform(.05, .95)))
        rho, b = float(rng.uniform(0, .95)), float(rng.uniform(0, 2))
        a = certified(**par, rho=rho, beta=b)
        e = certified(**{**par, 'P': par['P'] + b * par['B']}, rho=rho)
        err['completion_priority_equivalence'] = max(
            err['completion_priority_equivalence'],
            *(abs(a[k] - e[k]) for k in ('gain0', 'gain1', 'p', 't1', 't2', 'XQ', 'XE')))
        rr = float(np.exp(rng.uniform(-4, 0)))
        c1 = certified(**par, rho=rho, r=rr)
        c2 = certified(**par, rho=rho, r=rr, bonusD=b)
        err['release_bonus_shift'] = max(err['release_bonus_shift'],
                                         abs(c2['gain0'] - c1['gain0'] - b),
                                         abs(c2['gain1'] - c1['gain1'] - b),
                                         abs(c2['XQ'] - c1['XQ']), abs(c2['XE'] - c1['XE']))
        h = 1e-6
        dz = (certified(**{**par, 'P': par['P'] + h}, rho=rho, r=rr)['Z']
              - certified(**{**par, 'P': par['P'] - h}, rho=rho, r=rr)['Z']) / (2 * h)
        if dz <= 0:
            flags['dZdP_nonpositive'] += 1
    # Proposition 8: a loss contingent bonus lowers the hazard at every r
    flags['loss_bonus_not_decreasing'] = 0
    for _ in range(200):
        e1, e2 = np.exp(rng.uniform(-1.5, 1.5, 2))
        rr = float(np.exp(rng.uniform(-5, 1))) * (rng.uniform() < .8)
        rho = float(rng.uniform(0, .9))
        gs = np.linspace(0, (1 - rho) * 0.98, 30)
        Xs = [contest(e1, e2, 1.0, rho + g, 1 - rho, 0.0, rr)['X'] for g in gs]
        if np.any(np.diff(Xs) >= 0):
            flags['loss_bonus_not_decreasing'] += 1
    from powercost import hazard_attribution, solo_hazard
    # power costs: monotonicity for alpha >= 2 and location of minima below 2
    flags['power_nonmonotone_alpha_ge_2'] = 0
    argmins = []
    for _ in range(60):
        alpha = float(rng.choice([1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]))
        rr = float(np.exp(rng.uniform(-5, 0)))
        e1, e2 = np.exp(rng.uniform(-1.5, 1.5, 2))
        grid = np.linspace(0, 1, 81)
        Xs = [hazard_attribution(x, rr, alpha, e1, e2)[0] for x in grid]
        if alpha >= 2 and np.any(np.diff(Xs) > 1e-10):
            flags['power_nonmonotone_alpha_ge_2'] += 1
        if alpha < 2 and np.any(np.diff(Xs) > 1e-10):
            argmins.append(grid[int(np.argmin(Xs))])
    err['power_min_argmin_below_2'] = float(min(argmins)) if argmins else 1.0
    # power costs: homogeneity at r = 0 and the local condition near rho = 1
    from powercost import hazard_attribution, solo_hazard
    err['power_homogeneity_r0'] = 0.0
    flags['power_local_condition_mismatch'] = 0
    for _ in range(40):
        alpha = float(rng.uniform(1.2, 4.0))
        e1, e2 = np.exp(rng.uniform(-1.5, 1.5, 2))
        X0 = hazard_attribution(0.0, 0.0, alpha, e1, e2)[0]
        rho = float(rng.uniform(0.05, .95))
        X = hazard_attribution(rho, 0.0, alpha, e1, e2)[0]
        err['power_homogeneity_r0'] = max(err['power_homogeneity_r0'],
                                          abs(X / X0 - (1 - rho) ** (1 / (alpha - 1))))
        rr = float(np.exp(rng.uniform(-5, 1)))
        Xb = solo_hazard(rr, alpha, e1)
        cond = (alpha - 2) * Xb + (alpha - 1) * rr
        X1 = hazard_attribution(1.0, rr, alpha, e1, e2)[0]
        Xm = hazard_attribution(0.9995, rr, alpha, e1, e2)[0]
        if abs(cond) > 1e-3 and ((cond > 0) != (Xm > X1)):
            flags['power_local_condition_mismatch'] += 1
    return {'cases_contest': n, 'cases_certified': 150, 'seed': SEED,
            'maximum_residuals': {k: float(v) for k, v in err.items()},
            'violations': flags}


_C0CACHE = {}


def c0_ref(par):
    key = tuple(sorted(par.items()))
    if key not in _C0CACHE:
        c = certified(**par, rho=0.0)
        _C0CACHE[key] = {'QQ': (c['QQ']['X'], c['QQ']['cost']),
                         'EE': (c['EE']['X'], c['EE']['cost'])}
    return _C0CACHE[key]


def rho_D_fast(par):
    f = lambda x: certified(**par, rho=x)['gain1']
    if f(0) >= 0:
        return 0.0
    return brentq(f, 0, 0.999999, xtol=1e-13)


def monte_carlo(rng, draws=400000):
    """Simulate the certified game and compare mean T1 and T2 with formulas."""
    worst = 0.0
    for rho, r in [(0.0, 0.0), (0.2, 0.0), (0.3, 0.05), (0.6, 0.0), (0.9, 0.05)]:
        c = certified(**BASE, rho=rho, r=r)
        p, q = c['p'], BASE['q']
        u = rng.uniform(size=draws)
        both = u < c['pi11']
        rel = rng.uniform(size=(draws, 2)) < p
        T1 = np.zeros(draws)
        T2 = np.zeros(draws)
        # both ready
        nb = both.sum()
        d1, d2 = rel[both, 0], rel[both, 1]
        t2b = np.where(d1 & d2, rng.exponential(1 / c['XQ'], nb),
                       np.where(d1 ^ d2, rng.exponential(1 / c['XR'], nb),
                                rng.exponential(1 / c['XW'], nb)))
        t1b = np.where(~d1 & ~d2, t2b, 0.0)
        T1[both], T2[both] = t1b, t2b
        # one ready
        no = (~both).sum()
        d = rel[~both, 0]
        tr = rng.exponential(1 / c['xR'], no)
        ti = rng.exponential(1 / c['xI'], no)
        tpub = rng.exponential(1 / c['XR'], no)
        t1o = np.where(d, 0.0, np.minimum(tr, ti))
        t2o = np.where(d, rng.exponential(1 / c['XE'], no),
                       np.where(tr < ti, tr, ti + tpub))
        T1[~both], T2[~both] = t1o, t2o
        se1 = T1.std() / np.sqrt(draws)
        se2 = T2.std() / np.sqrt(draws)
        z1 = abs(T1.mean() - c['t1']) / max(se1, 1e-12) if c['t1'] > 0 else 0.0
        z2 = abs(T2.mean() - c['t2']) / se2
        worst = max(worst, z1, z2)
    return worst  # largest absolute z statistic


# ----------------------------------------------------------------------------
# Figure data
# ----------------------------------------------------------------------------
def fig_headline(nums):
    rD0, rD5 = nums['rho_D_r0'], nums['rho_D_r005']
    grid = np.unique(np.concatenate([np.linspace(0, .985, 400),
                                     [rD0, rD0 + 1e-7, rD5, rD5 + 1e-7]]))
    rows = []
    c00 = certified(**BASE, rho=0.0)
    for x in grid:
        a = certified(**BASE, rho=x)
        b = certified(**BASE, rho=x, r=0.05)
        rows.append([x, a['p'], a['t1'], a['t2'], frozen_t2(a, c00),
                     b['p'], b['t1'], b['t2']])
    write_csv('headline.csv', ['rho', 'p0', 't10', 't20', 't20frozen',
                               'p5', 't15', 't25'], rows)


def fig_regions(nums):
    # panel (a): equilibrium regions in the (rho, q) plane
    rhos = np.linspace(0, .98, 197)
    qs = np.linspace(.02, .98, 193)
    P = np.zeros((len(qs), len(rhos)))
    lo_line, hi_line = [], []
    for j, q in enumerate(qs):
        par = dict(BASE, q=q)
        g0 = np.array([certified(**par, rho=x)['gain0'] for x in rhos])
        g1 = np.array([certified(**par, rho=x)['gain1'] for x in rhos])
        p = np.clip(g0 / (g0 - g1), 0, 1)
        P[j] = p
    img = CMAP(P[::-1])
    plt.imsave(DATA / 'regions_heat.png', img)
    for q in np.linspace(.02, .98, 97):
        par = dict(BASE, q=q)
        f0 = lambda x: certified(**par, rho=x)['gain0']
        f1 = lambda x: certified(**par, rho=x)['gain1']
        a = 0.0 if f0(0) >= 0 else brentq(f0, 0, .9999)
        b = 0.0 if f1(0) >= 0 else brentq(f1, 0, .9999)
        lo_line.append([a, q])
        hi_line.append([b, q])
    write_csv('regions_withhold.csv', ['rho', 'q'], lo_line)
    write_csv('regions_release.csv', ['rho', 'q'], hi_line)
    qa, qb, xr = lo_line[0][1], lo_line[-1][1], 1.0
    write_csv('poly_withhold.csv', ['rho', 'q'],
              [[0.0, qa]] + lo_line + [[0.0, qb]])
    write_csv('poly_multiple.csv', ['rho', 'q'], lo_line + hi_line[::-1])
    write_csv('poly_release.csv', ['rho', 'q'],
              hi_line + [[xr, qb], [xr, qa]])
    nums['regions_q50_rho0'] = [r for r, q in lo_line if abs(q - .5) < 1e-9][0]
    # panel (b): technology map of rho_D over efficiency ratios
    le = np.linspace(np.log10(.5), np.log10(32), 101)
    li = np.linspace(np.log10(.01), np.log10(.5), 101)
    M = np.zeros((len(li), len(le)))
    for a, x in enumerate(li):
        for b, y in enumerate(le):
            M[a, b] = rho_D_fast(dict(BASE, etaE=10**y, etaI=10**x))
    nums['rhoD_map_min'], nums['rhoD_map_max'] = r4(M.min()), r4(M.max())
    plt.imsave(DATA / 'techmap_heat.png', CMAP(np.clip(M[::-1] / .5, 0, 1)))
    cs = plt.contour(le, li, M, levels=[.1, .2, .3, .4])
    for lev, seg in zip(cs.levels, cs.allsegs):
        pts = max(seg, key=len)
        write_csv(f'techmap_contour_{int(round(lev * 10))}.csv', ['le', 'li'], pts)
    plt.close('all')
    np.savetxt(DATA / 'techmap_baseline.csv',
               [[np.log10(BASE['etaE']), np.log10(BASE['etaI'])]],
               header='le li', comments='', fmt='%.6f')


def fig_impatience(nums):
    rs = [0.0, 0.01, 0.05, 0.2]
    grid = np.linspace(0, 1, 401)
    rows = []
    for x in grid:
        row = [x]
        for r in rs:
            if r == 0 and x > .999:
                row += [np.nan, np.nan]
                continue
            c = certified(**BASE, rho=x, r=r)
            row += [c['XE'], c['L']]
        rows.append(row)
    hdr = ['rho']
    for r in rs:
        tag = str(r).replace('.', 'p')
        hdr += [f'XE_{tag}', f'L_{tag}']
    write_csv('impatience.csv', hdr, rows)
    # relative slowdown under AI acceleration: r_eff = r/s with r = 0.05
    ss = np.logspace(0, 4, 41)
    out = []
    for s in ss:
        reff = 0.05 / s
        ta = certified(**BASE, rho=1.0, r=reff)['t2']
        tb = certified(**BASE, rho=0.5, r=reff)['t2']
        out.append([s, ta / tb, ta / s, tb / s])
    write_csv('acceleration.csv', ['s', 'ratio', 'T2full', 'T2half'], out)
    nums['accel_ratio_s1'] = r4(out[0][1])
    nums['accel_ratio_s100'] = r4(out[20][1])
    nums['accel_ratio_s10000'] = r4(out[-1][1])


def fig_design(nums):
    grid = np.linspace(0, 1.2, 241)
    rows = []
    for x in grid:
        a = certified(**BASE, rho=min(x, .995)) if x <= .995 else None
        b = certified(**BASE, beta=x)
        c = certified(**BASE, bonusD=x)
        rows.append([x, a['p'] if a else np.nan, a['t2'] if a else np.nan,
                     b['p'], b['t2'], c['p'], c['t2']])
    write_csv('design.csv', ['level', 'p_attr', 't2_attr', 'p_add', 't2_add',
                             'p_rel', 't2_rel'], rows)


def fig_welfare(nums):
    grid = np.linspace(0, .98, 393)
    rows = []
    for x in grid:
        c = certified(**BASE, rho=x)
        rows.append([x] + [welfare(c, K1=1.0, K2=K2, delta=0.1)
                           for K2 in (1.0, 2.0, 4.0)])
    write_csv('welfare.csv', ['rho', 'W1', 'W2', 'W4'], rows)
    for K2 in (1.0, 2.0, 4.0):
        f = lambda x: -welfare(certified(**BASE, rho=x), K1=1.0, K2=K2,
                               delta=0.1)
        lo = nums['rho_D_r0'] + 1e-9
        o = minimize_scalar(f, bounds=(lo, .98), method='bounded',
                            options={'xatol': 1e-10})
        best = min([(f(x), x) for x in np.linspace(lo, .98, 2000)])
        xs = o.x if f(o.x) <= best[0] else best[1]
        tag = f'K{int(K2)}'
        nums[f'welfare_opt_rho_{tag}'] = r4(xs)
        nums[f'welfare_opt_value_{tag}'] = r4(-f(xs))
        nums[f'welfare_rho0_{tag}'] = r4(-f(0.0))
        # bracket from the proposition
        c0 = certified(**BASE, rho=0.0)
        zs = [(np.sqrt(0.01 + K2 * 0.1 * c0[k]['X']**2 / c0[k]['cost']) - 0.1)
              / c0[k]['X'] for k in ('QQ', 'EE')]
        nums[f'welfare_zcirc_Q_{tag}'] = r4(zs[0])
        nums[f'welfare_zcirc_E_{tag}'] = r4(zs[1])
    with open(DATA / 'welfare_opt.csv', 'w') as fh:
        fh.write('rho W\n')
        for K2, tag in ((1.0, 'K1'), (2.0, 'K2'), (4.0, 'K4')):
            fh.write(f"{nums['welfare_opt_rho_' + tag]:.6f} "
                     f"{nums['welfare_opt_value_' + tag]:.6f}\n")


def w_release(z, K2, c0, delta=0.1, K1=1.0):
    """Welfare when every ready team releases and r = 0 (Proposition 6)."""
    tot = K1
    for key, pi in (('QQ', c0['pi11']), ('EE', c0['pi10'])):
        R, C = c0[key]['X'], c0[key]['cost']
        tot += pi * (K2 * R * z - C * z * z) / (delta + R * z)
    return tot


def beta_D_fast(rho, r=0.0):
    f = lambda b: min(certified(**BASE, rho=rho, beta=b, r=r)['gain1'],
                      certified(**BASE, rho=rho, beta=b, r=r)['gain0'])
    if f(0.0) > 0:
        return 0.0
    hi = 1.0
    while f(hi) <= 0:
        hi *= 2
    return brentq(f, 0, hi, xtol=1e-13)


def bonus_D(rho, r=0.0):
    """Least release bonus that makes all release the unique equilibrium."""
    c = certified(**BASE, rho=rho, r=r)
    return max(0.0, -min(c['gain0'], c['gain1']))


def prio_D(rho, r=0.0):
    """Least increase of priority credit P that secures release."""
    f = lambda b: min(certified(**{**BASE, 'P': BASE['P'] + b}, rho=rho, r=r)['gain0'],
                      certified(**{**BASE, 'P': BASE['P'] + b}, rho=rho, r=r)['gain1'])
    if f(0.0) > 0:
        return 0.0
    hi = 1.0
    while f(hi) <= 0:
        hi *= 2
    return brentq(f, 0, hi, xtol=1e-13)


def discount_factor(rho, delta=0.1, r=0.0, beta=0.0):
    c = certified(**BASE, rho=rho, r=r, beta=beta)
    return (c['pi11'] * c['XQ'] / (delta + c['XQ'])
            + c['pi10'] * c['XE'] / (delta + c['XE']))


N_REL = 2 / (2 - BASE['q'])   # expected number of releasing teams under all release


def outlay(rho, delta=0.1):
    """Expected outlay of the least release bonus that secures release."""
    return bonus_D(rho) * N_REL


def two_instruments(K2, rD, delta=0.1, net_grid=None):
    c0 = certified(**BASE, rho=0.0)
    zz1 = np.linspace(1e-6, 1, 40001)
    w1 = np.array([w_release(z, K2, c0, delta) for z in zz1])
    zs = zz1[int(np.argmax(w1))]
    o = minimize_scalar(lambda z: -w_release(z, K2, c0, delta),
                        bounds=(max(1e-9, zs - 1e-4), min(1, zs + 1e-4)),
                        method='bounded', options={'xatol': 1e-13})
    if w_release(o.x, K2, c0, delta) > w_release(zs, K2, c0, delta):
        zs = o.x
    rs = 1 - zs
    b = bonus_D(rs) if rs < rD else 0.0
    w2 = w_release(zs, K2, c0, delta)
    zz = np.linspace(1e-6, 1 - rD, 20001)
    wa = max(w_release(z, K2, c0, delta) for z in zz)
    out = b * N_REL
    zD, h = 1 - rD, 1e-7
    dW = (w_release(zD + h, K2, c0, delta) - w_release(zD, K2, c0, delta)) / h
    dg = (certified(**BASE, rho=rD)['gain1']
          - certified(**BASE, rho=rD - h)['gain1']) / h
    phim = dW / (N_REL * dg) if dW > 0 else 0.0
    res = dict(z=zs, rho=rs, bonus=b, W2=w2, Wattr=wa, outlay=out, phim=phim)
    if net_grid is not None:
        grid, O = net_grid

        def uses(phi):
            vals = [w_release(1 - x, K2, c0, delta) - phi * o_
                    for x, o_ in zip(grid, O)]
            return max(vals) > wa + 1e-12
        if not uses(1e-9):
            res['phihat'] = 0.0
        else:
            lo, hi = 1e-9, 1.0
            while uses(hi):
                hi *= 2
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if uses(mid) else (lo, mid)
            res['phihat'] = lo
    return res


def fig_twoinst(nums):
    rD = nums['rho_D_r0']
    grid = np.linspace(0, rD, 801)
    O = np.array([outlay(x) for x in grid])
    rows = []
    for K2 in np.linspace(1, 12, 111):
        t = two_instruments(K2, rD)
        rows.append([K2, t['rho'], t['bonus'], t['W2'] - t['Wattr'], t['phim']])
    write_csv('twoinst.csv', ['K2', 'rho', 'bonus', 'gain', 'phi'], rows)
    for K2 in (1.0, 2.0, 4.0, 8.0):
        t = two_instruments(K2, rD, net_grid=(grid, O))
        for k, v in t.items():
            nums[f'two_K{int(K2)}_{k}'] = v
        below = max(welfare(certified(**BASE, rho=x), K1=1.0, K2=K2, delta=0.1)
                    for x in np.linspace(0, rD - 1e-6, 400))
        nums[f'welfare_below_rhoD_K{int(K2)}'] = below
    # sensitivity of the welfare gain to the social discount rate and K2
    c0 = certified(**BASE, rho=0.0)
    for K2 in (4.0, 8.0, 16.0):
        for delta in (0.1, 0.5, 1.0):
            t = two_instruments(K2, rD, delta=delta)
            zz = np.linspace(1e-6, 1 - rD, 20001)
            za = zz[int(np.argmax([w_release(z, K2, c0, delta) for z in zz]))]
            tag = f'K{int(K2)}_d{str(delta).replace(".", "p")}'
            nums[f'sens_gain_pct_{tag}'] = 100 * (t['W2'] - t['Wattr']) / t['Wattr']
            nums[f'sens_t2red_pct_{tag}'] = 100 * (1 - za / t['z'])
            nums[f'sens_rho_{tag}'] = t['rho']
            nums[f'sens_phim_{tag}'] = t['phim']
    # implementation margins above rho_D
    for K2 in (4.0, 8.0):
        zD = 1 - rD
        for eps in (0.001, 0.01):
            loss = w_release(zD, K2, c0, 0.1) - w_release(zD - eps, K2, c0, 0.1)
            nums[f'impl_loss_K{int(K2)}_e{str(eps).replace(".", "p")}'] = loss
    # E[T2] of the asymmetric equilibrium (1,0) at rho_D
    c = certified(**BASE, rho=rD)
    RI = c['RI']
    nums['t2_asym_at_rhoD'] = (c['pi11'] / c['XR'] + 0.5 * c['pi10'] / c['XE']
                               + 0.5 * c['pi10'] * (1 / RI + (c['xI'] / RI) / c['XR']))


def instruments(nums):
    """Least level and outlay of each outside instrument that secures release."""
    rows = []
    for r in (0.0, 0.05):
        bD = bonus_D(0.0, r)
        bP = prio_D(0.0, r)
        bC = beta_D_fast(0.0, r)
        cR = certified(**BASE, rho=0.0, r=r, bonusD=bD + 1e-9)
        cP = certified(**{**BASE, 'P': BASE['P'] + bP + 1e-9}, rho=0.0, r=r)
        cC = certified(**BASE, rho=0.0, r=r, beta=bC + 1e-9)
        tag = str(r).replace('.', 'p')
        nums[f'inst_bonus_r{tag}'] = bD
        nums[f'inst_prio_r{tag}'] = bP
        nums[f'inst_comp_r{tag}'] = bC
        nums[f'inst_t2_bonus_r{tag}'] = cR['t2']
        nums[f'inst_t2_prio_r{tag}'] = cP['t2']
        nums[f'inst_t2_comp_r{tag}'] = cC['t2']
        for delta in (0.1, 0.5):
            dtag = str(delta).replace('.', 'p')
            nums[f'inst_out_bonus_r{tag}_d{dtag}'] = bD * N_REL
            nums[f'inst_out_prio_r{tag}_d{dtag}'] = bP
            nums[f'inst_out_comp_r{tag}_d{dtag}'] = bC * BASE['B'] * (
                cC['pi11'] * cC['XQ'] / (delta + cC['XQ'])
                + cC['pi10'] * cC['XE'] / (delta + cC['XE']))
    # social discount rate above which the completion bonus is cheaper than a
    # priority increase, for r = 0.05 and r = 0.2
    for r in (0.05, 0.2):
        bP, bC = prio_D(0.0, r), beta_D_fast(0.0, r)
        cC = certified(**BASE, rho=0.0, r=r, beta=bC + 1e-9)
        f = lambda d: bC * (cC['pi11'] * cC['XQ'] / (d + cC['XQ'])
                            + cC['pi10'] * cC['XE'] / (d + cC['XE'])) - bP
        nums[f'delta_star_r{str(r).replace(".", "p")}'] = brentq(f, 1e-6, 5)
    # marginal effect of priority credit on the value of withholding
    h = 1e-6
    nums['dZdP_rho0'] = (certified(**{**BASE, 'P': BASE['P'] + h}, rho=0.0)['Z']
                         - certified(**BASE, rho=0.0)['Z']) / h
    # loss contingent bonus: originator paid only when the rival completes
    for g in (0.0, 0.3, 0.6):
        z = contest(BASE['etaR'], BASE['etaR'], BASE['B'], g * BASE['B'],
                    BASE['B'], 0.0, 0.0)
        nums[f'loss_bonus_X_g{str(g).replace(".", "p")}'] = z['X']


def fig_power(nums):
    from powercost import hazard_attribution, solo_hazard
    rows = []
    grid = np.linspace(0, 1, 201)
    for x in grid:
        row = [x]
        for alpha in (1.5, 2.0, 3.0):
            X0 = hazard_attribution(0.0, 0.05, alpha)[0]
            row.append(hazard_attribution(x, 0.05, alpha)[0] / X0)
        for alpha in (1.5, 3.0):
            row.append((1 - x) ** (1 / (alpha - 1)))
        rows.append(row)
    write_csv('power.csv', ['rho', 'a15', 'a20', 'a30', 'z15', 'z30'], rows)
    for alpha in (1.5, 2.0, 3.0):
        X0 = hazard_attribution(0.0, 0.05, alpha)[0]
        tag = f'a{int(alpha * 10)}'
        for x in (0.9, 1.0):
            nums[f'power_{tag}_rho{int(x * 100)}'] = hazard_attribution(x, 0.05, alpha)[0] / X0
        Xb = solo_hazard(0.05, alpha)
        nums[f'power_{tag}_cond'] = (alpha - 2) * Xb + (alpha - 1) * 0.05
        vals = [hazard_attribution(x, 0.05, alpha)[0] for x in grid]
        i = int(np.argmin(vals))
        if 0 < i < len(grid) - 1:
            o = minimize_scalar(lambda x: hazard_attribution(x, 0.05, alpha)[0],
                                bounds=(grid[i - 1], grid[i + 1]), method='bounded',
                                options={'xatol': 1e-9})
            nums[f'power_{tag}_argmin'] = o.x
            nums[f'power_{tag}_min'] = o.fun / X0
        else:
            nums[f'power_{tag}_argmin'] = grid[i]


def fig_fixed(nums):
    rr = np.linspace(0, .7, 281)
    rows = []
    for x in rr:
        rows.append([x] + [fixed_game(q=q, rho=x)['p'] for q in (.3, .5, .85)])
    write_csv('fixed_attribution.csv', ['rho', 'q30', 'q50', 'q85'], rows)


# ----------------------------------------------------------------------------
# Numbers quoted in the text
# ----------------------------------------------------------------------------
def numbers():
    nums = {}
    c0 = certified(**BASE, rho=0.0)
    nums['rho_D_r0'] = rho_D_fast(BASE)
    nums['rho_U_r0'] = rho_U()
    nums['rho_D_r001'] = rho_D(r=0.01)
    nums['rho_D_r005'] = rho_D(r=0.05)
    nums['rho_D_r02'] = rho_D(r=0.2)
    nums['beta_D_r0'] = beta_D()
    for key in ['L', 'Z', 'D', 'gain0', 'gain1', 'Gamma', 'p', 't1', 't2',
                'xR', 'xI', 'XQ', 'XE', 'XW', 'RI', 'O', 'F', 'Q', 'C']:
        nums[f'base_rho0_{key}'] = c0[key]
    cD = certified(**BASE, rho=nums['rho_D_r0'] + 1e-9)
    nums['t2_at_rhoD'] = cD['t2']
    for x in (0.5, 0.9, 0.95, 0.99):
        c = certified(**BASE, rho=x)
        nums[f't2_rho{int(round(x * 100))}'] = c['t2']
    cb = certified(**BASE, beta=nums['beta_D_r0'] + 1e-9)
    nums['t2_additive_release'] = cb['t2']
    for r in (0.01, 0.05, 0.2):
        tag = str(r).replace('.', 'p')
        cf = certified(**BASE, rho=1.0, r=r)
        nums[f't2_full_r{tag}'] = cf['t2']
        nums[f'XQ_full_r{tag}'] = cf['XQ']
        nums[f'XE_full_r{tag}'] = cf['XE']
        cr = certified(**BASE, rho=0.0, r=r)
        nums[f't1_rho0_r{tag}'] = cr['t1']
        nums[f't2_rho0_r{tag}'] = cr['t2']
        grid = np.linspace(0, 1, 2001)
        Ls = [certified(**BASE, rho=x, r=r)['L'] for x in grid]
        i = int(np.argmax(Ls))
        o = minimize_scalar(lambda x: -certified(**BASE, rho=x, r=r)['L'],
                            bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]),
                            method='bounded', options={'xatol': 1e-10})
        nums[f'argmax_L_r{tag}'] = o.x
        nums[f'max_L_r{tag}'] = -o.fun
    nums['t2_rhoD_r0p05'] = certified(**BASE, rho=nums['rho_D_r005'] + 1e-7,
                                      r=0.05)['t2']
    # efficiency map anchors
    nums['rho_D_etaE8'] = rho_D_fast(dict(BASE, etaE=8.0))
    nums['rho_D_etaE2'] = rho_D_fast(dict(BASE, etaE=2.0))
    nums['rho_D_etaI02'] = rho_D_fast(dict(BASE, etaI=0.02))
    nums['rho_D_etaI1'] = rho_D_fast(dict(BASE, etaI=0.1))
    # fixed-intensity benchmark example
    g = fixed_game()
    nums['fixed_gain0'], nums['fixed_slope'], nums['fixed_p'] = g['d'], g['s'], g['p']
    # sponsor outlay of additive recognition (discount 0.1)
    b = nums['beta_D_r0']
    cb = certified(**BASE, beta=b + 1e-9)
    disc = (cb['pi11'] * cb['XQ'] / (0.1 + cb['XQ'])
            + cb['pi10'] * cb['XE'] / (0.1 + cb['XE']))
    nums['additive_outlay'] = b * BASE['B'] * disc
    return nums


def completion_counterexample(nums):
    """A completion bonus that induces release can delay the extension."""
    par = {**BASE, 'P': 0.02, 'q': 0.15, 'etaI': 0.4, 'etaE': 0.08}
    for beta in (0.01, 0.02):
        c = certified(**par, rho=0.0, beta=beta)
        tag = str(beta).replace('.', 'p')
        nums[f'cex_beta{tag}_p'] = c['p']
        nums[f'cex_beta{tag}_t2'] = c['t2']
        nums[f'cex_beta{tag}_gain0'] = c['gain0']
        nums[f'cex_beta{tag}_gain1'] = c['gain1']


def prize_topup(nums):
    """Welfare when an outside top up of B may push every race above z = 1."""
    c0 = certified(**BASE, rho=0.0)
    zz = np.linspace(1e-6, 3.0, 300001)
    w = np.array([w_release(z, 8.0, c0, 0.1) for z in zz])
    i = int(np.argmax(w))
    nums['topup_z_K8'] = zz[i]
    nums['topup_W_K8'] = w[i]


def global_counterexample(nums):
    """Constrained versus global welfare: withholding can beat assured release."""
    from model import welfare
    par = dict(P=0.1, q=0.5, etaI=0.0001, etaE=0.01)
    kw = dict(K1=0.001, K2=100.0, delta=20.0)
    rD = rho_D(**par)
    nums['cex_rhoD'] = rD
    nums['cex_W_rho0'] = welfare(certified(**{**BASE, **par}, rho=0.0), **kw)
    nums['cex_p_rho0'] = certified(**{**BASE, **par}, rho=0.0)['p']
    nums['cex_W_rhoD'] = welfare(certified(**{**BASE, **par}, rho=rD + 1e-9), **kw)


def main():
    nums = numbers()
    fig_headline(nums)
    fig_regions(nums)
    fig_impatience(nums)
    fig_design(nums)
    fig_welfare(nums)
    fig_twoinst(nums)
    instruments(nums)
    fig_power(nums)
    fig_fixed(nums)
    global_counterexample(nums)
    prize_topup(nums)
    completion_counterexample(nums)
    out = {k: (r4(v) if isinstance(v, (float, np.floating)) else v)
           for k, v in nums.items()}
    (RES / 'numbers.json').write_text(json.dumps(out, indent=2))
    ver = checks()
    (RES / 'verification.json').write_text(json.dumps(ver, indent=2))
    print(json.dumps(out, indent=2))
    print(json.dumps(ver, indent=2))


if __name__ == '__main__':
    main()
