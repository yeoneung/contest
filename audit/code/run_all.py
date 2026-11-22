#!/usr/bin/env python3
"""Run independent checks, failing on the first reported violation."""
from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
for script in ('independent_audit.py','power_cost_audit.py','deviation_audit.py','simulate_completion.py','boundary_audit.py','derivative_audit.py',
               'discounted_derivative_independent.py','fixed_budget_independent.py',
               'beath_mapping_audit.py','recycled_budget_independent.py','n_team_race_independent.py','ai_nonuniform_audit.py','budget_rule_local_independent.py'):
    print('Running',script,flush=True)
    subprocess.run([sys.executable,str(p/script)],check=True,stdout=subprocess.DEVNULL)
print('All specified numerical checks completed successfully.')
