#!/usr/bin/env python3
"""n-team quadratic Poisson race after disclosure; solves a scalar equation.

Winner i receives (1-rho)*B + rho*omega[i]*B + outside[i]; a loser
receives rho*omega[i]*B + outside[i], regardless of the winner's identity.
Only fixed post-release races are considered; release decisions are not solved.
"""
import numpy as np
from scipy.optimize import brentq

def contest(eta,B,rho,omega,r=0.,outside=None):
    eta=np.asarray(eta,dtype=float); omega=np.asarray(omega,dtype=float)
    outside=np.zeros_like(eta) if outside is None else np.asarray(outside,dtype=float)
    if (len(eta)<2 or eta.shape!=omega.shape or outside.shape!=eta.shape or
        np.any(eta<=0) or np.any(omega<0) or np.any(outside<0) or
        not np.isclose(omega.sum(),1.) or B<=0 or r<0 or not 0<=rho<=1):
        raise ValueError('Invalid contest parameters')
    if r==0 and rho==1: raise ValueError('Undiscounted zero-stake boundary is degenerate')
    d=(1-rho)*B; ell=rho*omega*B+outside; ed=eta*d
    c=ed**2-2*eta*r*ell
    lower=max(r,np.sqrt(max(0.,-float(np.min(c)))))
    def G(R):
        if r==0:
            return (len(eta)-1)-np.sum(R/(np.hypot(R,ed)+ed))
        S=np.sqrt(np.maximum(R*R+c,0.))
        # Rationalize sum of eta*d - sqrt to avoid cancellation.
        return r/R+(len(eta)-1)-np.sum((R*R-2*eta*r*ell)/(R*(S+ed)))
    lo=max(np.nextafter(lower,np.inf),1e-14)
    hi=max(1.,2*lo,2*max(ed))
    while G(hi)>0:hi*=2
    R=brentq(G,lo,hi,xtol=1e-14,rtol=1e-14)
    S=np.sqrt(np.maximum(R*R+c,0.))
    # Rationalization improves accuracy for small intensities.
    x=2*eta*(R*d+r*ell)/(R+ed+S)
    return dict(intensity=x,hazard=float(x.sum()),value=ell+d-x/eta,R=R)
