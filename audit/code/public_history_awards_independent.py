#!/usr/bin/env python3
"""Audit publication-conditioned winning awards with coupled best responses."""
from pathlib import Path
import hashlib
import json
from budget_region_map_independent import race, independent

ROOT=Path(__file__).resolve().parents[2]


def direct(p):
    P,B,q,er,ei,ee,tau=(p[k] for k in ['P','cap','q','etaR','etaI','etaE','tau'])
    low=B*(1-tau)
    qv,qf,qx=race(er,er,B,0.,B,0.)
    ev,ef,ex=race(er,ee,B,0.,low,0.)
    ov,of,_=race(er,er,B,0.,low,0.)
    av,af,_=race(er,er,low,0.,low,0.)
    zv,_,_=race(er,ei,P+low,af,P+av,0.)
    gain=[(1-q)*(P+ev-zv)+q*(P+ov-(P+low)/3),
          (1-q)*(P+ev-zv)+q*(P/2+qv-of)]
    x1,x2=er*(B-ev),ee*(low-ef)
    assert abs(x1+x2-ex)<1e-10
    qc=qx*(B-qv-qf);ec=x1*x1/(2*er)+x2*x2/(2*ee)
    wq,we=q/(2-q),2*(1-q)/(2-q)
    return dict(gain0=gain[0],gain1=gain[1],time_if_release=wq/qx+we/ex,
        welfare_if_release=1+wq*(8*qx-qc)/(.1+qx)+we*(8*ex-ec)/(.1+ex),
        expected_spending_if_release=wq*B+we*(x1*B+x2*low)/ex)


def main():
    saved=json.loads((ROOT/'results/public_history_awards.json').read_text())
    rows=[saved['baseline']]+saved['comparison_cases']
    errors=[]
    for row in rows:
        found=direct(row['parameters'])
        err=max(abs(found[k]-row[k])/(1+abs(row[k])) for k in found)
        assert err<1e-10,(row,err)
        assert row['unique_release']==(min(found['gain0'],found['gain1'])>0)
        assert 0<found['expected_spending_if_release']<=row['parameters']['cap']+1e-10
        errors.append(err)
    slopes=[independent(row) for row in saved['baseline_rediscovery_boundary']['bracket']]
    result=dict(passed=True,award_cases=len(rows),max_scaled_award_error=max(errors),
        input_sha256=hashlib.sha256((ROOT/'results/public_history_awards.json').read_bytes()).hexdigest(),
        boundary_slope_cases=len(slopes),
        max_scaled_boundary_slope_error=max(x['scaled_slope_error'] for x in slopes),
        method='Coupled best responses and separately constructed release-history payoffs. Production code is not imported; saved outputs supply targets only.')
    (ROOT/'audit/output/public_history_awards_independent.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
