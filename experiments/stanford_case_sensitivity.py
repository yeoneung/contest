#!/usr/bin/env python3
"""Audit the application's uncertainty bounds and contract comparators.

Keep three comparisons distinct: one rounded common contract, exact-threshold
contracts adapted to each scenario, and rounded scenario-specific contracts.
None of these model grids estimates the effect of the actual competition.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from model import certified
from recycled_budget_analysis import calculate
from stanford_award_study import activity, objects, publishable_share, CAP, BONUS_MAX

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/stanford_case_sensitivity'
LOWERS=(.001,.01,.05,.1)


def scenarios(floor, lower):
    return [(float(q),float(i),float(e),float(p))
            for q in np.linspace(floor,.99,5)
            for i in np.geomspace(lower,4.,5)
            for e in np.geomspace(.25,16.,5)
            for p in (.01,.1,.2)]


def summary(rows, key):
    vals=np.array([r[key] for r in rows])
    return dict(cases=len(rows),minimum=float(vals.min()),median=float(np.median(vals)),
                maximum=float(vals.max()),faster=int(np.sum(vals<1-1e-9)),
                slower=int(np.sum(vals>1+1e-9)))


def evaluate(floor, lower):
    states=scenarios(floor,lower);funcs=[objects(*s) for s in states]
    b=BONUS_MAX/CAP
    roots={'A':[f[0](0.,'reservation') for f in funcs],
           'F':[f[0](b,'reservation') for f in funcs],
           'R':[f[0](b,'recycling') for f in funcs]}
    shares={k:publishable_share(max(v)) for k,v in roots.items()}
    driving={}
    for k,v in roots.items():
        idx=int(np.argmax(v));q,i,e,p=states[idx]
        driving[k]=dict(q=q,rediscovery=i,method_use=e,priority_to_cap=p,
                        exact_threshold=v[idx],maximizers=int(np.sum(np.abs(np.array(v)-max(v))<1e-11)))
    rows=[]
    for ix,(state,func) in enumerate(zip(states,funcs)):
        q,i,e,p=state;thr,time=func
        base=time(0.,shares['A'],'reservation')
        row=dict(q=q,mI=i,mE=e,priority_ratio=p,
                 rho_A=roots['A'][ix],rho_F=roots['F'][ix],rho_R=roots['R'][ix],
                 common_F_ratio=time(b,shares['F'],'reservation')/base,
                 common_R_ratio=time(b,shares['R'],'recycling')/base,
                 positive_initial_threshold=roots['A'][ix]>1e-10)
        if row['positive_initial_threshold']:
            for rounded,prefix in ((False,'threshold'),(True,'rounded')):
                a,f,r=(roots[k][ix] for k in ('A','F','R'))
                if rounded:a,f,r=map(publishable_share,(a,f,r))
                base_i=time(0.,a,'reservation')
                row[prefix+'_F_ratio']=time(b,f,'reservation')/base_i
                row[prefix+'_R_ratio']=time(b,r,'recycling')/base_i
        rows.append(row)
    positive=[r for r in rows if r['positive_initial_threshold']]
    return dict(readiness_floor=floor,rediscovery_lower=lower,scenario_count=len(states),
                common_shares=shares,driving_scenarios=driving,
                common_F=summary(rows,'common_F_ratio'),common_R=summary(rows,'common_R_ratio'),
                scenario_specific={k:summary(positive,k+'_ratio') for k in ['threshold_F','threshold_R','rounded_F','rounded_R']},
                winning_increment_usd={'A':CAP*(1-shares['A']),
                                       'F':(CAP-2*BONUS_MAX)*(1-shares['F']),
                                       'R_two_releases':(CAP-2*BONUS_MAX)*(1-shares['R']),
                                       'R_one_release':(CAP-BONUS_MAX)*(1-shares['R'])},
                rows=rows)


def independent(q,i,e,p,b,rho,rule):
    if rule=='reservation':
        c=certified(P=p,B=1-2*b,etaR=1.,etaI=i,etaE=e,q=q,rho=rho,bonusD=b)
        time=c['pi11']/c['QQ']['X']+c['pi10']/c['EE']['X']
    else:
        c=calculate(dict(P=p,K=1.,etaR=1.,etaI=i,etaE=e,q=q),b,rho)
        time=c['t2']
    return float(time),min(c['gain0'],c['gain1'])


def audit(results):
    errors=[];strict=[];boundary=[]
    b=BONUS_MAX/CAP
    for result in results:
        shares=result['common_shares']
        for row in result['rows']:
            args=(row['q'],row['mI'],row['mE'],row['priority_ratio'])
            _,time=objects(*args)
            for k,rule,bb in [('A','reservation',0.),('F','reservation',b),('R','recycling',b)]:
                t,g=independent(*args,bb,shares[k],rule)
                errors.append(abs(t-time(bb,shares[k],rule)));strict.append(g)
            if row['positive_initial_threshold']:
                for k,rule,bb in [('A','reservation',0.),('F','reservation',b),('R','recycling',b)]:
                    rho=row['rho_'+k]
                    t,g=independent(*args,bb,rho,rule)
                    errors.append(abs(t-time(bb,rho,rule)));boundary.append(g)
                    _,strict_gain=independent(*args,bb,rho+1e-7,rule)
                    strict.append(strict_gain)
    assert max(errors)<1e-9 and min(strict)>0 and min(boundary)>-1e-10
    return dict(passed=True,continuation_time_comparisons=len(errors),
                strict_release_checks=len(strict),max_time_discrepancy=max(errors),
                minimum_strict_release_gain=min(strict),minimum_boundary_gain=min(boundary),
                independent_method='General contest and release-history solvers, compared with the closed-form sensitivity implementation.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    case=activity();floor=next(r['readiness_floor'] for r in case['readiness_floors'] if r['validation_TM_threshold']==.3)
    bounds=[]
    for lower in LOWERS:
        row=evaluate(floor,lower);bounds.append(row)
        print('rediscovery lower bound',lower,'F',row['common_F']['median'],
              'R',row['common_R']['minimum'],row['common_R']['maximum'],flush=True)
    screening=[evaluate(q,.001) for q in (.5,.8)]
    checks=audit(bounds+screening)
    # Preserve every paired setting, including rounding exceptions.
    fields=list(bounds[0]['rows'][0])
    fields+=list(dict.fromkeys(k for result in bounds+screening for row in result['rows'] for k in row if k not in fields))
    with (OUT/'paired_settings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['exercise','rediscovery_lower']+fields);writer.writeheader()
        for exercise,results in [('bounds',bounds),('screening',screening)]:
            for result in results:
                for row in result['rows']:
                    writer.writerow(dict(exercise=exercise,rediscovery_lower=result['rediscovery_lower'],**row))
    output=dict(published_bonus_usd=BONUS_MAX,design_cap_usd=CAP,
                bounds=bounds,screening=screening,checks=checks,
                comparison_notes={'common':'One rounded contract for the whole uncertainty grid, compared with one rounded no-bonus contract on the same grid.',
                                  'threshold':'Scenario-specific exact release thresholds; limits of strict-release contracts.',
                                  'rounded':'Scenario-specific shares rounded upward by 0.1 percentage point. Small recycling exceptions are retained, not discarded.',
                                  'scope':'Directional results concern this fixed payment and these finite grids, not every bonus or every parameter.'},
                paired_csv_sha256=hashlib.sha256((OUT/'paired_settings.csv').read_bytes()).hexdigest())
    (OUT/'sensitivity_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'checks':checks,'primary_scenario_specific':bounds[0]['scenario_specific']},indent=2))


if __name__=='__main__':main()
