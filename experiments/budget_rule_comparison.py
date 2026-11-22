#!/usr/bin/env python3
"""Compare budget accounting rules analytically and numerically (not data).

Local slopes follow binding all-release constraints, rather than a selected
mixed equilibrium that can jump at numerical threshold errors.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq
from model import certified, k_ratio
from common_budget_global import analysis
from recycled_budget_analysis import search, calculate

ROOT = Path(__file__).resolve().parent.parent


def local_comparison(par):
    P,K,eR,eI,eE,q=(par[k] for k in ('P','K','etaR','etaI','etaE','q'))
    args=dict(P=P,etaR=eR,etaI=eI,etaE=eE,q=q,r=0.)
    def D(B,rho): return certified(**args,B=B,rho=rho)['gain1']
    rho0=brentq(lambda rho:D(K,rho),0.,.999999,xtol=1e-14)
    dR=P+(2+rho0)*K/3; dI=P+(1+2*rho0)*K/3
    k=k_ratio(eR*dR/(eI*dI));ke=k_ratio(eR/eE)
    eps=2+k/(k+2)-2*k/(2*k+1)
    kr=k/eps*((K/3)/dR-(2*K/3)/dI)
    kb=k/eps*(((2+rho0)/3)/dR-((1+2*rho0)/3)/dI)
    Dr=(1-q)*(2*K/(ke+2)+2*(K/3)/(k+2)-2*dR*kr/(k+2)**2)+q*K/2
    DB=(1-q)*(-2*(1-rho0)/(ke+2)+2*((2+rho0)/3)/(k+2)-2*dR*kb/(k+2)**2)+q*rho0/2
    H=2*(1-q)/(ke+2)+2*q/3
    c=certified(**args,B=1.,rho=0.)
    pi=np.array([c['pi11'],c['pi10']]);h=np.array([c['QQ']['X'],c['EE']['X']])
    S0=K*(1-rho0);T0=float(np.sum(pi/h)/S0)
    SFprime=K*(1-2*DB)/Dr-2*(1-rho0)
    TFprime=-T0*SFprime/S0
    TRprime=float((2*pi[0]/h[0]+pi[1]/h[1])/(K*K*(1-rho0))-T0*H/Dr)
    fd=[]
    for b in (1e-4,1e-5,1e-6):
        rhoF=brentq(lambda rho:D(K-2*b,rho)+b,0.,.999999,xtol=1e-14)
        rhoR=brentq(lambda rho:D(K,rho)+b*(1-rho)*H,0.,.999999,xtol=1e-14)
        TF=float(np.sum(pi/h)/((K-2*b)*(1-rhoF)))
        TR=calculate(par,b,rhoR)['t2']
        fd.append(dict(b=b,fixed=(TF-T0)/b,recycling=(TR-T0)/b))
    assert abs(fd[-1]['fixed']-TFprime)<2e-5
    assert abs(fd[-1]['recycling']-TRprime)<2e-5
    return dict(rho0=rho0,T0=T0,D_B=DB,D_rho=Dr,H=H,
                fixed_stake_derivative=SFprime,fixed_time_derivative=TFprime,
                recycling_time_derivative=TRprime,one_sided_finite_differences=fd)


if __name__=='__main__':
    par=dict(P=.2,K=1.,etaR=1.,etaI=.05,etaE=4.,q=.5)
    local=local_comparison(par)
    fixed=analysis(dict(P=par['P'],cap=par['K'],etaR=par['etaR'],etaI=par['etaI'],etaE=par['etaE'],q=par['q']))
    recycled=search(par)
    w0=fixed['welfare_table'][-1]['welfare_attribution']
    w1=fixed['welfare_table'][-1]['welfare_budget_design']
    result=dict(parameters=par,local=local,
                fixed_numeric=dict(b=fixed['bmax_numeric'],rho=fixed['rho_at_bmax'],
                    time=fixed['expected_time_at_Smax'],welfare=w1),
                recycling_numeric=recycled['fastest_numeric'],
                recycling_welfare_numeric=recycled['welfare_numeric'][-1],
                speed_gain_percent=100*(1-fixed['expected_time_at_Smax']/local['T0']),
                welfare_gain=w1-w0,welfare_gain_percent=100*(w1/w0-1),
                note='Local derivatives are analytical. Designs use numerical grid/refinement, not a global certificate. Threshold outcomes are all-release limits.')
    (ROOT/'results'/'budget_rule_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
