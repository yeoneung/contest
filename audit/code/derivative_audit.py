#!/usr/bin/env python3
"""Check the left derivative at full attribution against direct best-response equilibria."""
import importlib.util,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('direct_solver',P/'code/independent_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(84822);worst=0.;checks=0
for _ in range(400):
 eta1,eta2,B=10**rng.uniform(-1,1,3);r=10**rng.uniform(-2,1);h=1e-7
 full=m.paramrace(eta1,eta2,B,1.,r,1.);near=m.paramrace(eta1,eta2,B,1-h,r,1.)
 dx=(full[2]-near[2])/h; X=2*eta1*r*B/(math.sqrt(r*r+2*eta1*r*B)+r)
 expected=-(eta2/eta1)*B*X/(X+r);err=abs(dx-expected)/max(abs(expected),1e-9)
 worst=max(worst,err);checks+=1
 if err>1e-3:raise AssertionError((eta1,eta2,B,r,dx,expected,err))
out={'seed':84822,'draws':checks,'finite_difference_step':h,'max_relative_error':worst,'limitations':'One-sided finite differences test this boundary identity numerically, not its complete proof.'}
(P/'output/derivatives.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
