"""Two-team Poisson race with power costs c_i(x) = x^alpha / (alpha * eta_i).

Stationary equilibrium conditions (any discount rate r >= 0):
    (r + x_j) x_i^(alpha-1)/eta_i + (1 - 1/alpha) x_i^alpha/eta_i = r w_i + x_j d_i.
The left side is strictly increasing in x_i, so each condition defines a unique
best response psi_i(x_j). Equilibria are fixed points of psi_1 o psi_2.
"""
import numpy as np
from scipy.optimize import brentq


def best_response(xj, eta, w, l, r, alpha):
    d = w - l
    rhs = r * w + xj * d
    if rhs <= 0:
        return 0.0
    f = lambda x: ((r + xj) * x**(alpha - 1) + (1 - 1 / alpha) * x**alpha) / eta - rhs
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
    return brentq(f, 0.0, hi, xtol=1e-15, rtol=1e-14)


def race(eta1, eta2, w1, l1, w2, l2, r, alpha, grid=400):
    """Return all stationary equilibria (x1, x2) found by scanning x1."""
    def phi(x1):
        x2 = best_response(x1, eta2, w2, l2, r, alpha)
        return x1 - best_response(x2, eta1, w1, l1, r, alpha), x2
    # upper bound for x1: best response to a very aggressive rival or solo
    top = 2 * max(best_response(0.0, eta1, w1, l1, r, alpha),
                  best_response(1e3, eta1, w1, l1, r, alpha), 1e-6) + 1.0
    xs = np.linspace(1e-12, top, grid)
    vals = [phi(x)[0] for x in xs]
    sols = []
    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if fa == 0:
            sols.append(a)
        elif fa * fb < 0:
            sols.append(brentq(lambda x: phi(x)[0], a, b, xtol=1e-15))
    out = []
    for x1 in sols:
        x2 = phi(x1)[1]
        out.append((x1, x2))
    if not out and abs(vals[0]) < 1e-10:
        out.append((0.0, phi(0.0)[1]))
    return out


def hazard_attribution(rho, r, alpha, eta1=1.0, eta2=1.0, B=1.0, omega=1.0):
    """Total hazard of the race after release under attribution share rho."""
    w1, l1 = (1 - rho) * B + rho * omega * B, rho * omega * B
    w2, l2 = (1 - rho) * B + rho * (1 - omega) * B, rho * (1 - omega) * B
    eq = race(eta1, eta2, w1, l1, w2, l2, r, alpha)
    return [x1 + x2 for x1, x2 in eq]


def solo_hazard(r, alpha, eta=1.0, B=1.0):
    return best_response(0.0, eta, B, B, r, alpha)
