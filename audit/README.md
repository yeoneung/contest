# Numerical audit

From the archive root (the directory containing `build_figures.py`), install
the dependencies in `requirements.txt` and run:

```
python audit/code/run_all.py
python audit/code/common_budget_global_audit.py
python audit/code/budget_region_map_independent.py
python audit/code/discounted_budget_independent.py
python audit/code/recycling_profile_independent.py
python audit/code/contract_extensions_independent.py
python audit/code/accounting_strength_independent.py
python audit/code/public_history_awards_independent.py
```

The first command runs 13 scripts covering direct best responses, cost curvature,
boundary cases, numerical effort deviations, event simulations, discounted
derivatives, fixed-budget checks, payoff mappings, recycled bonuses, and local
budget-accounting effects. It includes 350 coupled best-response checks for the
many-team post-release race and 1,000 checks of nonuniform efficiency effects.
Results are written to `audit/output/`. The second command compares the
common-budget stake and welfare reductions against the general race solver.
It imports `experiments/model.py` and `experiments/common_budget_global.py`, so it is a
cross-check rather than a fully independent implementation.

The third command checks the budget-region maps using independently solved
coupled best responses and separate payoff tables for release histories. It
checks all 102 both-accelerate grid points, 180 seeded draws from other interior
regions, and 36 zero-attribution release cases. The maximum scaled slope error
in the reported run is below 1.12e-7. It imports no production slope, release-gain,
or scalar-hazard functions.

The discounted audit checks 325 points from the 5,043-setting positive-discount
grid and six baseline cases varying discounting and certification timing. It
uses direct coupled best responses, independently constructed history-specific
payoffs, fresh release-boundary solves, and right-sided Richardson derivatives.
The maximum scaled slope discrepancy is below 3.78e-8. It imports no production
contest solver or slope function.

The recycling-profile audit checks 121 parameter cases: 106 feasible and 15
infeasible. Feasible cases include 69 lower-endpoint, 36 interior, and one
upper-limit bonus optimum. Independent best-response welfare and a separately
optimized bonus agree with the analytical inner optimizer within 2.69e-11 in
scaled welfare. This validates sampled calculations, not the numerical outer
search globally.

Each test covers a finite set of parameter values and numerical deviations. It does not establish a theorem or rule out unexamined equilibria. All retained analytical proofs are in the single article, including the unrestricted-deviation verification and continuation-value proof in Appendix A. Detailed outside-funding welfare and instrument results are omitted from the compact article; auxiliary numerical diagnostics remain in this archive. The scripts use fixed random seeds and require no outside data.

The contract-extension audit compares 160 penalty and reconsideration cases
with coupled best-response calculations and independent election payoff
tables. “Independent” denotes a separate mathematical implementation, not
independent authors, AI tools, or an external review. The reconsideration
checks apply to one additional election, not unrestricted disclosure timing.

The accounting-strength audit checks the new analytical reversal region and
effect magnitudes. It uses the coupled-best-response implementation, separately
constructs release-history payoffs, and solves fresh threshold roots. It checks
80 local-slope cases at rediscovery ratios 1e-4 and 1e-8, 39 magnitude settings,
663 alternative feasible bonuses, and the same-bonus baseline under both rules.
Maximum scaled slope and magnitude errors are 1.38e-6 and 5.22e-15. Comparison
targets are read from saved outputs; no production model or optimization
function enters the audit calculation. The mathematical proof, including
convergence from positive rediscovery efficiency, is in Appendix A of the
article. These finite checks are not a global optimization certificate.

The public-history award audit checks 82 cases and two local slopes around the
baseline rediscovery boundary. It keeps certification for play and separately
constructs payout histories using only initial public release and completion.
The maximum scaled award and slope errors are 4.14e-16 and 6.95e-9. Both new
audits record hashes of their saved numerical inputs.
