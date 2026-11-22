#!/usr/bin/env python3
"""Redraw the case decision figure from saved applied-study results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-grid',action='store_true',help='Use saved case results (the default).')
    parser.parse_args()
    saved=json.loads((ROOT/'results/stanford_case_sensitivity/sensitivity_results.json').read_text())
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.85))
    colors={'reservation':'#2166ac','recycling':'#c65d15'}
    x=np.array([r['rediscovery_lower'] for r in saved['bounds']])
    f=np.array([r['common_F']['median'] for r in saved['bounds']])
    rlo=np.array([r['common_R']['minimum'] for r in saved['bounds']])
    rhi=np.array([r['common_R']['maximum'] for r in saved['bounds']])
    axes[0].plot(x,f,color=colors['reservation'],lw=1.8,marker='o',ms=3.5,label='Reservation')
    axes[0].plot(x,rhi,color=colors['recycling'],lw=1.8,marker='o',ms=3.5,label='Recycling')
    axes[0].plot(x,rlo,color=colors['recycling'],lw=.6)
    axes[0].fill_between(x,rlo,rhi,color=colors['recycling'],alpha=.16,lw=0)
    axes[0].set_xscale('log')
    axes[0].set_xticks(x,labels=['0.001','0.01','0.05','0.1'])
    axes[0].set(title='(a) Common contract: bound sensitivity',
                xlabel=r'Assumed lower bound on $\eta_I/\eta_R$',ylabel='Time / no-bonus time')
    axes[0].axhline(1,color='0.35',lw=.7,linestyle='--',zorder=0)
    rows=[r for r in saved['bounds'][0]['rows'] if r['positive_initial_threshold']]
    all_values=[]
    for name,key in [('reservation','threshold_F_ratio'),('recycling','threshold_R_ratio')]:
        vals=np.sort([r[key] for r in rows]);all_values.extend(vals)
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',lw=1.8,color=colors[name])
    axes[1].set(title='(b) Scenario-specific threshold contracts',
                xlabel='Time / no-bonus time',ylabel='Cumulative fraction of grid settings',ylim=(0,1.02))
    axes[1].axvline(1,color='0.35',lw=.7,linestyle='--',zorder=0)
    axes[1].set_xlim(min(all_values)-.025,max(all_values)+.025)
    for ax in axes:
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.5,1.035),ncol=2,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.96),w_pad=1.4)
    output=ROOT/'figures/fig_stanford_decision.pdf'
    fig.savefig(output,bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Saved',output)


if __name__=='__main__':
    main()
