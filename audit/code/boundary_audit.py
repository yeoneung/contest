#!/usr/bin/env python3
"""Check the full-attribution and zero-discount boundaries."""
import importlib.util,json,math
from pathlib import Path
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('direct_solver',P/'code/independent_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
e1,e2,B=1.,4.,1.;rows=[]
for r in (1e-12,1e-9,1e-6,1e-3,.1,1.,100.):
 x,y,*_=m.paramrace(e1,e2,B,1.,r,1.)
 expected=2*e1*r*B/(math.sqrt(r*r+2*e1*r*B)+r)
 assert abs(x+y-expected)<1e-11*max(1,expected)
 rows.append({'r':r,'full_share_hazard':x+y,'formula':expected})
interior=m.paramrace(e1,e2,B,.5,0.,1.)
near=m.paramrace(e1,e2,B,.5,1e-12,1.)
assert abs(interior[0]+interior[1]-near[0]-near[1])<1e-8
out={'full_share_positive_discount':rows,'interior_r0_hazard':interior[0]+interior[1],'interior_r_1e-12_hazard':near[0]+near[1], 'r0_rho1':'No stationary equilibrium in the sole-originator race: follower optimally chooses zero; against a zero rival the originator has an unattained supremum as x approaches zero from above.'}
(P/'output/boundaries.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
