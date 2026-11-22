#!/usr/bin/env python3
"""Public-record case audit and financially constrained award comparison.

The observed financial limits are distinct from model scenarios. Filtered,
selected submission records give documented-readiness floors, not estimates
of private readiness, research costs, or causal policy effects. Scores are
retrospective evaluations, not a reconstruction of a historical leaderboard.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from model import k_ratio, certified
from recycled_budget_analysis import calculate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'stanford_rna'
OUT = ROOT / 'results' / 'stanford_award_study'
CAP = 35000.0
BONUS_MAX = 2500.0


def date(value):
    return datetime.strptime(value[:10], '%m/%d/%Y').date()


def activity():
    teams = list(csv.DictReader((DATA / 'teams.csv').open()))
    submissions = list(csv.DictReader((DATA / 'submissions.csv').open()))
    versions = {r['Id']: r for r in csv.DictReader((DATA / 'kernelversions.csv').open())}
    leaders = {r['Id']: r['TeamLeaderId'] for r in teams}
    eligible = {r['Id'] for r in teams if r['CompetitionId'] == '87793'
                and r['IsBenchmark'] == 'False' and r['PrivateLeaderboardRank']}
    retained = [r for r in submissions if r['TeamId'] in eligible
                and r['IsAfterDeadline'] == 'False']
    cutoff = datetime(2025, 4, 23).date()
    deadline = datetime(2025, 5, 29).date()
    records = []
    for threshold in (.2, .3, .4):
        demonstrators = {r['TeamId'] for r in retained
                        if r['SourceKernelVersionId'] in versions
                        and versions[r['SourceKernelVersionId']]['AuthorUserId'] == leaders[r['TeamId']]
                        and date(versions[r['SourceKernelVersionId']]['CreationDate']) < cutoff
                        and r['PrivateScoreFullPrecision']
                        and float(r['PrivateScoreFullPrecision']) >= threshold}
        records.append(dict(validation_TM_threshold=threshold,
                            documented_ready_teams=len(demonstrators),
                            cohort_teams=len(eligible),
                            readiness_floor=len(demonstrators)/len(eligible)))
    all_case = [r for r in submissions if r['TeamId'] in eligible]
    return dict(competition_id=87793, snapshot=2401,
                retained_case_teams=sum(r['CompetitionId']=='87793' for r in teams),
                final_ranked_nonbenchmark_teams=len(eligible),
                retained_submission_rows=len(all_case),
                predeadline_flag_submission_rows=len(retained),
                selected_flag_counts={x:sum(r['IsSelected']==x for r in all_case) for x in ('True','False','')},
                reevaluation_dated_rows=sum(r['SubmissionDate']=='09/24/2025' for r in retained),
                source_versions_matched=sum(r['SourceKernelVersionId'] in versions for r in retained),
                leader_authored_source_versions_matched=sum(r['SourceKernelVersionId'] in versions and versions[r['SourceKernelVersionId']]['AuthorUserId']==leaders[r['TeamId']] for r in retained),
                decision_date=str(cutoff), submission_deadline=str(deadline),
                days_to_deadline=(deadline-cutoff).days,
                readiness_floors=records,
                interpretation='For the retrospective pool of final-ranked teams, a qualifying source version authored by its final team leader and created before April 23 documents that the leader already held a working predictor. With the additional planning assumption that this demonstrates possession of a usable method and that two candidates are drawn independently from this pool, the fraction supplies a conservative readiness floor. Unmatched, later, and non-leader-authored versions do not establish non-readiness. Thresholds are analyst-specified, not the official VFOLD sharing criterion. SubmissionDate is a re-evaluation date and is not used as the creation clock. This does not estimate q or research efficiency ratios.')


def objects(q, mI, mE, priority):
    ke = k_ratio(1/mE)
    pi = np.array([q/(2-q), 2*(1-q)/(2-q)])
    h = np.array([4/3, 2/(ke+2)+2*ke*mE/(2*ke+1)])
    H = 2*(1-q)/(ke+2)+2*q/3

    def gain(B, rho):
        dR=priority+(2+rho)*B/3
        dI=priority+(1+2*rho)*B/3
        k=k_ratio(dR/(mI*dI))
        L=priority+B*(rho+(1-rho)*ke/(ke+2))
        Z=priority+B-2*dR/(k+2)
        return (1-q)*(L-Z)+q*(priority+rho*B)/2

    def threshold(b, rule):
        def d(rho):
            return gain(1-2*b,rho)+b if rule=='reservation' else gain(1,rho)+b*(1-rho)*H
        return 0. if d(0)>=0 else brentq(d,0.,1.,xtol=1e-13)

    def time(b, rho, rule):
        stakes=(1-2*b)*(1-rho)*np.ones(2) if rule=='reservation' else np.array([1-2*b,1-b])*(1-rho)
        return float(np.sum(pi/(h*stakes)))

    return threshold, time


def publishable_share(required):
    """Round up to a 0.1-percentage-point share with a strict margin."""
    return float(np.ceil((required+1e-8)*1000)/1000)


def study(floor, grid_size=5, bonus_steps=51):
    # These technology and recognition ranges are disclosed sensitivity assumptions.
    # No runtime, submission count, or score is converted into a research-cost estimate.
    qs=np.linspace(max(floor,1e-5),.99,grid_size)
    mIs=np.geomspace(.001,4.,grid_size)
    mEs=np.geomspace(.25,16.,grid_size)
    priorities=(.01,.1,.2)
    scenarios=[(float(q),float(i),float(e),float(p)) for q in qs for i in mIs for e in mEs for p in priorities]
    functions=[objects(*s) for s in scenarios]
    bmax=BONUS_MAX/CAP
    rho0=publishable_share(max(f[0](0.,'reservation') for f in functions))
    base=np.array([f[1](0.,rho0,'reservation') for f in functions])
    rows=[]
    for rule in ('reservation','recycling'):
        def evaluate(b):
            rho=publishable_share(max(f[0](float(b),rule) for f in functions))
            ratios=np.array([f[1](float(b),rho,rule) for f in functions])/base
            payouts=[75000-2*b*CAP*(1-q)/(2-q) if rule=='reservation' else 75000 for q in qs]
            return dict(rule=rule,bonus_usd=float(b*CAP),rho_uniform=float(rho),
                        min_time_ratio=float(ratios.min()),max_time_ratio=float(ratios.max()),
                        minimum_expected_awards_usd=float(min(payouts)),
                        maximum_expected_awards_usd=float(max(payouts)))
        scan=[evaluate(float(b)) for b in np.linspace(0,bmax,bonus_steps)]
        best=min(scan,key=lambda r:r['max_time_ratio'])
        rows.append(dict(rule=rule,published_bonus=scan[-1],minimax_relative_time=best,scan=scan))
    return dict(readiness_floor=floor,q_lower_grid=float(qs[0]),q_upper_scenario=.99,scenarios=len(scenarios),grid_size=grid_size,
                technology_scenarios={'rediscovery_ratio':[.001,4.], 'method_use_ratio':[.25,16.],
                                      'recognition_to_cap':[.01,.1,.2]},
                no_bonus_uniform_attribution=rho0,policies=rows,
                bonus_menu_usd=list(np.linspace(0.,BONUS_MAX,bonus_steps)),
                attribution_increment=.001,
                objective='Minimize the largest expected-time ratio relative to one no-bonus attribution rule that induces release throughout the same finite scenario set. Attribution is one common share across scenarios, rounded upward in 0.1-percentage-point increments to give strict incentives. All 51 bonus levels ($50 increments) are enumerated for both accounting rules. This is the best policy on the specified finite scenario set and bonus menu, not an optimum over a continuous uncertainty box or unrestricted contracts.')


def audit(case, studies):
    errors=[]
    release_checks=0
    smallest_gain=float('inf')
    for record in studies:
        floor=record['readiness_floor']
        for q in (max(floor,1e-5),.5,.99):
            for mI,mE,P in ((.001,16.,.01),(.05,4.,.1),(1.,1.,.2)):
                f,t=objects(q,mI,mE,P)
                for rule in ('reservation','recycling'):
                    b=BONUS_MAX/CAP;rho=f(b,rule)
                    par=dict(P=P,K=1.,etaR=1.,etaI=mI,etaE=mE,q=q)
                    if rule=='reservation':
                        c=certified(P=P,B=1-2*b,etaR=1,etaI=mI,etaE=mE,q=q,rho=rho,bonusD=b)
                        gain=c['gain1'];tt=c['pi11']/c['QQ']['X']+c['pi10']/c['EE']['X']
                    else:
                        c=calculate(par,b,rho);gain=c['gain1'];tt=c['t2']
                    assert gain > -1e-10
                    errors.append(abs(tt-t(b,rho,rule)))
        for policy in record['policies']:
            for row in (policy['minimax_relative_time'],policy['published_bonus']):
                b=row['bonus_usd']/CAP;rho=row['rho_uniform']
                gains=[]
                for q in np.linspace(max(floor,1e-5),.99,record['grid_size']):
                    for mI in np.geomspace(.001,4.,record['grid_size']):
                        for mE in np.geomspace(.25,16.,record['grid_size']):
                            for P in (.01,.1,.2):
                                par=dict(P=P,K=1.,etaR=1.,etaI=mI,etaE=mE,q=q)
                                if policy['rule']=='reservation':
                                    c=certified(P=P,B=1-2*b,etaR=1,etaI=mI,etaE=mE,q=q,rho=rho,bonusD=b)
                                else:c=calculate(par,b,rho)
                                g=min(c['gain0'],c['gain1'])
                                assert g>0
                                gains.append(g*CAP);release_checks+=1
                row['minimum_release_gain_usd']=float(min(gains))
                smallest_gain=min(smallest_gain,min(gains))
    assert max(errors)<1e-9
    return dict(passed=True,independent_continuation_comparisons=len(errors),
                maximum_time_discrepancy=max(errors),attribution_rounding_increment=.001,
                independent_strict_release_checks=release_checks,
                smallest_release_gain_usd=smallest_gain,
                financial_reconciliation={'total_usd':75000,'protected_placement_payments_usd':40000,
                                          'design_cap_usd':35000,'two_sharing_awards_usd':5000,
                                          'residual_winning_stake_usd':30000})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    case=activity()
    (OUT/'activity_audit.json').write_text(json.dumps(case,indent=2)+'\n')
    studies=[]
    for row in case['readiness_floors']:
        s=study(row['readiness_floor']);s['validation_TM_threshold']=row['validation_TM_threshold'];studies.append(s)
        print('completed threshold',row['validation_TM_threshold'],flush=True)
    screening=[]
    for floor in (.5,.8):
        s=study(floor);s['interpretation']='Hypothetical lower bound established by a future candidate-screening exercise; not an observed readiness rate.'
        screening.append(s)
        print('completed proposed screening floor',floor,flush=True)
    checks=audit(case,studies+screening)
    result=dict(case=case,studies=studies,proposed_screening_sensitivity=screening,checks=checks,
                source_hashes={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['teams.csv','submissions.csv','kernelversions.csv']})
    (OUT/'case_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'case':case,'checks':checks,'policies':[{k:v for k,v in s.items() if k in ['readiness_floor','validation_TM_threshold','no_bonus_uniform_attribution','scenarios']}|{'policies':[{k:v for k,v in x.items() if k!='scan'} for x in s['policies']]} for s in studies]},indent=2))


if __name__=='__main__':
    main()
