"""Equilibrium objects for the certified release game with endogenous effort.

All formulas follow the manuscript. The extension contest is solved for any
private discount rate r >= 0 through the scalar total-hazard equation
    G(R) = [r + R + sum_i eta_i d_i - sum_i sqrt(R^2 + c_i)] / R = 0,
    c_i = eta_i^2 d_i^2 - 2 eta_i r l_i,
whose left side is strictly decreasing (Theorem 1 of the paper).
"""
import numpy as np
from scipy.optimize import brentq


def contest(eta1, eta2, w1, l1, w2, l2, r=0.0):
    """Stationary Markov equilibrium of the two-player Poisson contest.

    Player i chooses intensity x_i, pays x_i^2/(2 eta_i) per unit time and
    receives w_i after its own success and l_i after the rival's success.
    Returns intensities, values, total hazard R - r and aggregate flow cost.
    """
    d1, d2 = w1 - l1, w2 - l2
    assert d1 >= 0 and d2 >= 0 and l1 >= 0 and l2 >= 0
    c1 = eta1**2 * d1**2 - 2 * eta1 * r * l1
    c2 = eta2**2 * d2**2 - 2 * eta2 * r * l2

    def g(R):
        return (r + R + eta1 * d1 + eta2 * d2
                - np.sqrt(max(R * R + c1, 0.0)) - np.sqrt(max(R * R + c2, 0.0)))

    def G(R):
        if r == 0:
            # Rationalize eta*d - sqrt(R**2 + (eta*d)**2).  Direct
            # subtraction cancels to zero near the lower bracket for large
            # stakes, sometimes giving Brent's method the wrong sign.
            s1, s2 = eta1*d1, eta2*d2
            return 1.0 - R/(s1 + np.hypot(R, s1)) - R/(s2 + np.hypot(R, s2))
        return g(R) / R

    if r == 0 and (d1 == 0 or d2 == 0):
        raise ValueError('Undiscounted race with a zero winning stake is degenerate; use the appropriate limit.')
    lo = max(r, np.sqrt(max(0.0, -c1)), np.sqrt(max(0.0, -c2)))
    lo = max(lo, 1e-14)
    if G(lo) <= 0:  # only possible through rounding at r > 0
        lo = lo * (1 + 1e-9) + 1e-14
    hi = max(1.0, 2 * lo)
    while G(hi) > 0:
        hi *= 2
    R = brentq(G, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=500)
    x1 = R + eta1 * d1 - np.sqrt(max(R * R + c1, 0.0))
    x2 = R + eta2 * d2 - np.sqrt(max(R * R + c2, 0.0))
    V1 = w1 - x1 / eta1
    V2 = w2 - x2 / eta2
    cost = x1 * x1 / (2 * eta1) + x2 * x2 / (2 * eta2)
    return dict(x1=x1, x2=x2, V1=V1, V2=V2, X=x1 + x2, cost=cost, R=R)


def h(k):
    return k * k * (k + 2) / (2 * k + 1)


def k_ratio(theta):
    """Unique positive root of h(k) = theta."""
    return brentq(lambda k: h(k) - theta, 1e-15, 2 * np.sqrt(theta) + 2,
                  xtol=1e-15, rtol=1e-14)


def certified(P=0.2, B=1.0, etaR=1.0, etaI=0.05, etaE=4.0, q=0.5, rho=0.0,
              r=0.0, beta=0.0, bonusD=0.0):
    """All continuation objects of the certified commitment game.

    rho  : redistributive attribution, a share rho of the extension pool B is
           paid to the registered original publisher(s) whoever wins.
    beta : completion bonus, the registered original publisher(s) receive
           beta*B (split equally if simultaneous) when the first extension is
           published, financed outside the pool.
    bonusD : release bonus paid by a sponsor at date zero to each ready team
           that releases its method immediately.
    """
    bB = beta * B
    # sole originator vs independently ready follower
    RR = contest(etaR, etaR, B + bB, rho * B + bB, (1 - rho) * B, 0.0, r)
    O, F = RR['V1'], RR['V2']
    # simultaneous original publication
    QQ = contest(etaR, etaR, (1 - rho / 2) * B + bB / 2, rho * B / 2 + bB / 2,
                 (1 - rho / 2) * B + bB / 2, rho * B / 2 + bB / 2, r)
    Q = QQ['V1']
    # sole originator vs exposed (enabled) rival
    EE = contest(etaR, etaE, B + bB, rho * B + bB, (1 - rho) * B, 0.0, r)
    L = P + EE['V1'] + bonusD
    # mutual withholding: the winner publishes both results
    WW = contest(etaR, etaR, P + B + bB, 0.0, P + B + bB, 0.0, r)
    C = WW['V1']
    # private contest: withholder (extension) vs unready rival (discovery)
    II = contest(etaR, etaI, P + B + bB, F, P + O, 0.0, r)
    Z = II['V1']
    X, Y = P + O + bonusD, P / 2 + Q + bonusD
    D = L - Z
    gain0 = (1 - q) * D + q * (X - C)      # against a withholding rival
    gain1 = (1 - q) * D + q * (Y - F)      # against a disclosing rival
    Gamma = gain0 - gain1
    if gain1 >= 0 and gain0 >= 0:
        p = 1.0
    elif gain0 <= 0 and gain1 <= 0:
        p = 0.0
    else:
        p = float(np.clip(gain0 / Gamma, 0, 1))
    pi11, pi10 = q / (2 - q), 2 * (1 - q) / (2 - q)
    xR, xI = II['x1'], II['x2']
    RI = II['X']
    t1 = pi10 * (1 - p) / RI + pi11 * (1 - p) ** 2 / WW['X']
    t2 = (pi11 * (p * p / QQ['X'] + 2 * p * (1 - p) / RR['X']
                  + (1 - p) ** 2 / WW['X'])
          + pi10 * (p / EE['X'] + (1 - p) * (1 / RI + (xI / RI) / RR['X'])))
    return dict(O=O, F=F, Q=Q, L=L, C=C, Z=Z, X=X, Y=Y, D=D,
                gain0=gain0, gain1=gain1, Gamma=Gamma, p=p, t1=t1, t2=t2,
                xR=xR, xI=xI, RI=RI, XR=RR['X'], XQ=QQ['X'], XE=EE['X'],
                XW=WW['X'], RR=RR, QQ=QQ, EE=EE, WW=WW, II=II,
                pi11=pi11, pi10=pi10, beta=beta, B=B)


def welfare(c, K1=1.0, K2=1.0, delta=0.05):
    """Expected discounted social surplus under the symmetric selection."""
    p, pi11, pi10 = c['p'], c['pi11'], c['pi10']

    def stage2(cc):
        return (K2 * cc['X'] - cc['cost']) / (delta + cc['X'])

    wQ, wR, wE = stage2(c['QQ']), stage2(c['RR']), stage2(c['EE'])
    WW = c['WW']
    wW = ((K1 + K2) * WW['X'] - WW['cost']) / (delta + WW['X'])
    II = c['II']
    wI = (II['x1'] * (K1 + K2) + II['x2'] * (K1 + wR) - II['cost']) / (
        delta + II['X'])
    both = p * p * (K1 + wQ) + 2 * p * (1 - p) * (K1 + wR) + (1 - p) ** 2 * wW
    one = p * (K1 + wE) + (1 - p) * wI
    return pi11 * both + pi10 * one
