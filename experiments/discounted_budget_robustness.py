#!/usr/bin/env python3
"""Discounted local accounting effects using history-specific continuation races.

Both release incentive endpoints are checked. The study follows the boundary
of the high-attribution all-release region, not a mixed-equilibrium selection.
"""
from pathlib import Path
from functools import lru_cache
from collections import Counter
import argparse
import csv
import json
import platform
import numpy as np
import scipy
from scipy.optimize import brentq
from model import certified
from budget_region_map import local_effects, draw_panel, COLORS, NAMES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]


class Accounting:
    def __init__(self, P=.2, q=.5, etaI=.05, etaE=4., r=.05, public=False):
        self.P, self.q, self.etaI, self.etaE, self.r = P,q,etaI,etaE,r
        self.public=public
        self.state=lru_cache(maxsize=512)(self._state)

    def _state(self, B, rho):
        return certified(P=self.P,B=B,etaR=1.,etaI=self.etaI,etaE=self.etaE,
                         q=self.q,rho=rho,r=self.r)

    def outcome(self, b, rho, rule):
        if rule=='F':
            c0=c1=c2=self.state(1-2*b,rho)
        else:
            c0=self.state(1.,rho);c1=self.state(1-b,rho);c2=self.state(1-2*b,rho)
        solo=c1['L']+b-c0['Z']
        ready_w=c1['X']+b-c0['C']
        ready_d=c2['Y']+b-c1['F']
        gains=np.array([solo,ready_w,ready_d] if self.public else
                       [(1-self.q)*solo+self.q*ready_w,
                        (1-self.q)*solo+self.q*ready_d])
        T=self.q/(2-self.q)/c2['QQ']['X']+2*(1-self.q)/(2-self.q)/c1['EE']['X']
        return gains,float(T)

    def threshold(self,b=0.,rule='F'):
        top=1. if self.r>0 else 1-1e-8
        grid=np.linspace(0.,top,17)
        vals=np.array([min(self.outcome(b,float(z),rule)[0]) for z in grid])
        crossings=[i for i in range(16) if vals[i]<=0<vals[i+1]]
        if vals[0]>1e-10:
            return 0.,len(crossings),float(vals[0])
        if not crossings:
            raise RuntimeError(('No release boundary',self.__dict__,vals))
        i=crossings[-1]
        rho=brentq(lambda z:min(self.outcome(b,z,rule)[0]),grid[i],grid[i+1],
                   xtol=2e-13,rtol=2e-13)
        return float(rho),len(crossings),float(vals[0])

    def local(self):
        rho,nroots,gain_zero=self.threshold()
        empty=dict(rhoD=rho,T0=np.nan,slope_fixed=np.nan,slope_recycling=np.nan,
                   active_constraint=-1,constraint_gap=np.nan,derivative_stability=np.nan,
                   sampled_upcrossings=nroots,gain_at_zero=gain_zero)
        if rho==0:
            return dict(empty,region=0)
        g0,T0=self.outcome(0.,rho,'F')
        active=int(np.argmin(g0));gap=float(np.partition(g0,1)[1]-g0[active])
        if gap<1e-8:
            return dict(empty,T0=T0,active_constraint=active,constraint_gap=gap,region=5)
        hr=min(1e-4,rho/8,(1-rho)/8)
        def derivatives(rule,hb,hr):
            base=np.r_[self.outcome(0.,rho,rule)[0][active],T0]
            def vector(b,z):
                gains,T=self.outcome(float(b),float(z),rule)
                return np.array([gains[active],T])
            # Second-order right derivative in b; fourth-order central derivative in rho.
            db=(-3*base+4*vector(hb,rho)-vector(2*hb,rho))/(2*hb)
            dr1=(vector(0,rho+hr)-vector(0,rho-hr))/(2*hr)
            dr2=(vector(0,rho+hr/2)-vector(0,rho-hr/2))/hr
            dr=(4*dr2-dr1)/3
            if dr[0]<=0:
                raise RuntimeError(('Nonregular boundary',rho,dr,self.__dict__))
            return float((db[1]-dr[1]*db[0]/dr[0])/T0)
        coarse=np.array([derivatives(j,1e-4,hr) for j in ['F','R']])
        fine=np.array([derivatives(j,5e-5,hr/2) for j in ['F','R']])
        error=float(np.max(abs(fine-coarse)/(1+abs(fine))))
        if error>2e-6:
            raise RuntimeError(('Unstable derivatives',rho,coarse,fine,error))
        sf,sr=map(float,fine)
        region=5 if min(abs(sf),abs(sr))<1e-6 else (1 if sr<0 else 2) if sf<0 else (3 if sr<0 else 4)
        return dict(rhoD=rho,T0=T0,slope_fixed=sf,slope_recycling=sr,
                    active_constraint=active,constraint_gap=gap,derivative_stability=error,
                    sampled_upcrossings=nroots,gain_at_zero=gain_zero,region=region)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=41)
    parser.add_argument('--reuse-grid',action='store_true')
    args=parser.parse_args()
    gi=np.geomspace(.001,10,args.points);ge=np.geomspace(.1,100,args.points)
    data=ROOT/'figures/data/discounted_budget_map.csv';rows=[]
    if args.reuse_grid:
        with data.open() as f:
            for row in csv.DictReader(f):
                rows.append({k:int(v) if k in ['region','region_r0','active_constraint','sampled_upcrossings'] else float(v) for k,v in row.items()})
        assert len(rows)==3*args.points**2
    else:
        for q in [.2,.5,.8]:
            for e in ge:
                for i in gi:
                    obj=Accounting(q=q,etaI=float(i),etaE=float(e))
                    row=dict(priority_ratio=.2,q=q,mI=float(i),mE=float(e),r=.05,
                             **obj.local(),region_r0=local_effects(.2,q,float(i),float(e))['region'])
                    rows.append(row)
            print('Computed discounted grid q=',q,flush=True)
        with data.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    baseline=[]
    for r in [0.,.05,.2]:
        for public in [False,True]:
            c=Accounting(r=r,public=public);v=c.local()
            baseline.append(dict(r=r,certification_before_election=public,**v,
                                 derivative_fixed=v['slope_fixed']*v['T0'],
                                 derivative_recycling=v['slope_recycling']*v['T0']))
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
                         'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(7,5.3),sharex=True,sharey=True)
    for col,q in enumerate([.2,.5,.8]):
        panel=[r for r in rows if r['q']==q]
        for row,rate in enumerate([0,.05]):
            selected=[dict(r,region=r['region_r0']) for r in panel] if row==0 else panel
            draw_panel(axs[row,col],selected,gi,ge,q,.2)
            axs[row,col].set_title(rf'$r={rate:g},\ q={q:g}$',fontsize=10)
            if row==1:axs[row,col].set_xlabel(r'Rediscovery efficiency $\eta_I/\eta_R$')
            if col==0:axs[row,col].set_ylabel(r'Method-use efficiency $\eta_E/\eta_R$')
    present=sorted({r[k] for r in rows for k in ['region','region_r0']})
    fig.legend(handles=[Patch(facecolor=COLORS[k],label=NAMES[k]) for k in present],
               loc='lower center',ncol=2,frameon=False,fontsize=8,labelspacing=.4)
    fig.subplots_adjust(left=.08,right=.99,top=.95,bottom=.18,hspace=.22,wspace=.18)
    fig.savefig(ROOT/'figures/fig_discounted_budget.pdf',dpi=600,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(ROOT/'figures/fig_discounted_budget.png',dpi=180);plt.close(fig)
    counts=Counter(str(r['region']) for r in rows)
    report=dict(total_settings=len(rows),grid_points_per_axis=args.points,
                parameters=dict(P=.2,cap=1.,etaR=1.,r=.05),regions=NAMES,region_counts=dict(counts),
                classification_agreement_with_r0=sum(r['region']==r['region_r0'] for r in rows),
                regular_interior_settings=sum(r['region'] in [1,2,3,4] for r in rows),
                multiple_sampled_upcrossings=sum(r['sampled_upcrossings']>1 for r in rows),
                max_derivative_stability=max(r['derivative_stability'] for r in rows if np.isfinite(r['derivative_stability'])),
                baseline_and_timing=baseline,
                method='History-specific discounted continuation races; both private-information incentive endpoints; 17-point bracketing of the high-attribution release region; implicit-boundary slopes with second-order right differences in bonus and Richardson central differences in attribution.',
                normalization='cap=eta_R=1; slopes=(cap/T0)dT/db; r=.05 is relative to eta_R*cap.',
                software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    (ROOT/'results/discounted_budget_robustness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
