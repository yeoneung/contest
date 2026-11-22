#!/usr/bin/env python3
"""Reproduce local budget-accounting maps from the undiscounted closed forms.

The map follows a binding all-release incentive constraint. It compares marginal
bonuses under the same ex post cap, with priority credit fixed outside that cap.
It is a model sensitivity analysis; no experimental data enter these figures.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import platform
import numpy as np
import scipy
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from model import k_ratio

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    0: 'Release already strictly optimal at zero attribution',
    1: 'Both rules accelerate',
    2: 'Reservation accelerates; recycling slows',
    3: 'Reservation slows; recycling accelerates',
    4: 'Both rules slow',
    5: 'Zero-slope or zero-gain boundary',
}
COLORS = ['#e6e8eb', '#52a594', '#347fc4', '#dfa24d', '#bc7165', '#ffffff']


def local_effects(priority_ratio, q, mI, mE):
    """Return normalized slopes (cap/T0)dT/db, with cap=eta_R=1."""
    P = float(priority_ratio)
    ke = k_ratio(1 / mE)

    def gain(rho):
        dr = P + (2 + rho) / 3
        di = P + (1 + 2 * rho) / 3
        k = k_ratio(dr / (mI * di))
        return (1-q)*(-2*(1-rho)/(ke+2)+2*dr/(k+2))+q*(P+rho)/2

    d0 = gain(0.)
    empty = dict(rhoD=np.nan, T0=np.nan, D_B=np.nan, D_rho=np.nan,
                 slope_fixed=np.nan, slope_recycling=np.nan)
    if d0 > 1e-11:
        return dict(**empty, gain_at_zero=d0, region=0)
    if abs(d0) <= 1e-11:
        return dict(**empty, gain_at_zero=d0, region=5)
    rho = brentq(gain, 0., 1., xtol=1e-13, rtol=1e-13)
    dr = P + (2+rho)/3
    di = P + (1+2*rho)/3
    k = k_ratio(dr/(mI*di))
    elasticity = 2+k/(k+2)-2*k/(2*k+1)
    kr = k/elasticity*((1/3)/dr-(2/3)/di)
    kb = k/elasticity*(((2+rho)/3)/dr-((1+2*rho)/3)/di)
    Dr = (1-q)*(2/(ke+2)+2/(3*(k+2))-2*dr*kr/(k+2)**2)+q/2
    DB = (1-q)*(-2*(1-rho)/(ke+2)+2*((2+rho)/3)/(k+2)
                -2*dr*kb/(k+2)**2)+q*rho/2
    H = 2*(1-q)/(ke+2)+2*q/3
    hQ = 4/3
    hE = 2/(ke+2)+2*ke*mE/(2*ke+1)
    piQ, piE = q/(2-q), 2*(1-q)/(2-q)
    A = piQ/hQ+piE/hE
    T0 = A/(1-rho)
    sf = 2-(1-2*DB)/((1-rho)*Dr)
    sr = (2*piQ/hQ+piE/hE)/A-H/Dr
    if min(abs(sf), abs(sr)) <= 1e-10:
        region = 5
    elif sf < 0:
        region = 1 if sr < 0 else 2
    else:
        region = 3 if sr < 0 else 4
    return dict(rhoD=rho, T0=T0, D_B=DB, D_rho=Dr,
                slope_fixed=sf, slope_recycling=sr,
                gain_at_zero=d0, region=region)


def draw_panel(ax, rows, grid_i, grid_e, q, priority_ratio):
    region = np.array([r['region'] for r in rows]).reshape(len(grid_e),len(grid_i))
    ax.pcolormesh(grid_i,grid_e,region,cmap=ListedColormap(COLORS),
                  norm=BoundaryNorm(np.arange(-.5,6.5),6),shading='nearest',
                  edgecolors='none',antialiased=False,rasterized=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(grid_i[0],grid_i[-1]); ax.set_ylim(grid_e[0],grid_e[-1])
    ax.set_xticks([.001,.01,.1,1,10]);ax.set_yticks([.1,1,10,100])
    ax.tick_params(which='major',length=3,pad=2)
    ax.tick_params(which='minor',length=1.5)
    for spine in ax.spines.values():spine.set_color('#788390');spine.set_linewidth(.5)
    if abs(q-.5)<1e-12 and abs(priority_ratio-.2)<1e-12:
        ax.plot(.05,4,marker='D',markersize=4,color='#d88c1a',
                markeredgecolor='white',markeredgewidth=.6)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=81)
    parser.add_argument('--reuse-grid',action='store_true',
                        help='Redraw the existing grid CSV without recalculating its values.')
    args=parser.parse_args()
    if args.points<3:parser.error('--points must be at least 3')
    grid_i=np.geomspace(.001,10,args.points)
    grid_e=np.geomspace(.1,100,args.points)
    priorities=[.05,.2,.5];qs=[.2,.5,.8]
    panels={};allrows=[];counts=[];cached={}
    if args.reuse_grid:
        with (ROOT/'figures/data/budget_region_map.csv').open() as f:
            for row in csv.DictReader(f):
                row={k:(int(v) if k=='region' else float(v)) for k,v in row.items()}
                cached.setdefault((row['priority_ratio'],row['q']),[]).append(row)
    for priority in priorities:
        for q in qs:
            if args.reuse_grid:
                rows=cached[(priority,q)]
                assert len(rows)==args.points**2,'Cached grid has a different resolution.'
                assert np.allclose([r['mI'] for r in rows],np.tile(grid_i,args.points))
                assert np.allclose([r['mE'] for r in rows],np.repeat(grid_e,args.points))
            else:
                rows=[]
                for mE in grid_e:
                    for mI in grid_i:
                        row=dict(priority_ratio=priority,q=q,mI=float(mI),mE=float(mE),
                                 **local_effects(priority,q,float(mI),float(mE)))
                        rows.append(row)
            panels[(priority,q)]=rows;allrows.extend(rows)
            counts.append(dict(priority_ratio=priority,q=q,points=len(rows),
                               regions={NAMES[k]:v for k,v in sorted(Counter(r['region'] for r in rows).items())}))
            action='Loaded' if args.reuse_grid else 'Computed'
            print(f'{action} P/cap={priority:g}, q={q:g}: {len(rows)} settings',flush=True)
    data=ROOT/'figures/data/budget_region_map.csv'
    data.parent.mkdir(parents=True,exist_ok=True)
    with data.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':9,'axes.titlesize':10,'axes.labelsize':10,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    present=sorted({r['region'] for r in allrows})
    legend=[Patch(facecolor=COLORS[k],label=NAMES[k]) for k in present]
    fig,axs=plt.subplots(1,3,figsize=(7.0,3.15),sharey=True)
    for n,(ax,q) in enumerate(zip(axs,qs)):
        draw_panel(ax,panels[(.2,q)],grid_i,grid_e,q,.2)
        ax.set_title(f'({chr(97+n)}) $q={q:g}$')
        ax.set_xlabel(r'Rediscovery efficiency $\eta_I/\eta_R$')
    axs[0].set_ylabel(r'Method-use efficiency $\eta_E/\eta_R$')
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,0.0),
               ncol=1,frameon=False,fontsize=8.5,handlelength=1.3,labelspacing=.3)
    fig.subplots_adjust(left=.08,right=.99,top=.9,bottom=.34,wspace=.2)
    fig.savefig(ROOT/'figures/fig_budget_regions.pdf',dpi=600,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(ROOT/'figures/fig_budget_regions.png',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(7.0,7.25),sharex=True,sharey=True)
    for row,priority in enumerate(priorities):
        for col,q in enumerate(qs):
            ax=axs[row,col];draw_panel(ax,panels[(priority,q)],grid_i,grid_e,q,priority)
            ax.set_title(rf'$P/\mathcal{{B}}={priority:g},\ q={q:g}$',fontsize=9.5)
            if row==2:ax.set_xlabel(r'$\eta_I/\eta_R$')
            if col==0:ax.set_ylabel(r'$\eta_E/\eta_R$')
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.0),ncol=1,
               frameon=False,fontsize=9,handlelength=1.3,labelspacing=.3)
    fig.subplots_adjust(left=.08,right=.99,top=.96,bottom=.15,hspace=.3,wspace=.18)
    fig.savefig(ROOT/'figures/fig_budget_regions_sensitivity.pdf',dpi=600,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    result={'grid_points_per_axis':args.points,'total_settings':len(allrows),
            'software':{'python':platform.python_version(),'numpy':np.__version__,
                        'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
            'priority_to_cap_ratios':priorities,'readiness_probabilities':qs,
            'rediscovery_ratio_range':[float(grid_i[0]),float(grid_i[-1])],
            'method_use_ratio_range':[float(grid_e[0]),float(grid_e[-1])],
            'normalization':'eta_R=cap=1; r=beta=0; normalized slope=(cap/T0)*dT/db',
            'scope':'Local derivatives along binding all-release constraints; no empirical data or population frequencies.',
            'regions':NAMES,'panels':counts,'baseline':local_effects(.2,.5,.05,4.)}
    (ROOT/'results/budget_region_map.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_settings':len(allrows),'regions_present':[NAMES[k] for k in present],
                      'baseline':result['baseline']},indent=2))


if __name__=='__main__':main()
