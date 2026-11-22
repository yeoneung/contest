# Replication: Rewarding Disclosure in Research Contests

Junkee Jeon and Yeoneung Kim.

This repository contains model calculations, a public-data case study, figure
sources, and numerical checks for research contests with disclosure incentives.
It compares attribution, release payments, and the treatment of unused bonus
funds. Observed award constraints, derived capability bounds, and assumed
technology scenarios are distinguished throughout the case study.

## Quick start

Clone the repository and run the commands from its root:

```sh
git clone https://github.com/yeoneung/contest.git
cd contest
```

Use Python 3.9 or later. All paths below are relative to this directory.

```sh
python -m pip install -r requirements.txt
python experiments/stanford_award_study.py
python experiments/stanford_case_sensitivity.py
python build_figures.py
```

The case calculation uses the supplied small data subsets and requires no network.
Figure 1 requires Tectonic or pdfLaTeX; Figures 2–3 use Python. Set `TECTONIC`
to the executable path if necessary. Tectonic obtains missing packages on its
first run; `TECTONIC_OFFLINE=1` uses the cached resources. The numerical package
versions used for the reference calculations are pinned in `requirements-reproduction.txt`.

```text
experiments/          Model, public-data extraction, and numerical studies
figures/              Editable figure sources and generated graphics
figures/data/         Saved numerical grid inputs
results/              Case results, grids, and model calculations
data/stanford_rna/    Versioned small case subsets, source hashes, and linkage notes
audit/code/           Independent numerical verification implementations
audit/output/         Saved verification results
```

## Figure and table index

| Output | Calculation or source | Saved inputs/results |
|---|---|---|
| Figure 1: release workflow | `figures/fig_timeline.tex` | Static TikZ; `figures/figpreamble.tex` |
| Figure 2: Stanford award choices | `experiments/plot_stanford_awards.py` | `results/stanford_case_sensitivity/sensitivity_results.json` |
| Figure 3: accounting-effect regions | `experiments/budget_region_map.py` | `figures/data/budget_region_map.csv`; 59,049 settings |
| Table 1: illustrative baseline | `experiments/reproduce.py` | `results/numbers.json` |
| Table 2: capped contracts and outcomes | `experiments/budget_rule_comparison.py`, `common_budget_global.py`, `recycled_budget_analysis.py` | Corresponding `results/` JSON files |
| Table 3: case inputs | `data/institutional_awards.csv`, `data/stanford_rna/`, `experiments/stanford_award_study.py` | Actual financial limits and separately labeled scenario inputs |
| Table 4: rediscovery-bound sensitivity | `experiments/stanford_case_sensitivity.py` | `results/stanford_case_sensitivity/sensitivity_results.json` |
| Table 5: discounting and certification | `experiments/discounted_budget_robustness.py` | `results/discounted_budget_robustness.json` |
| Table C.1: broader award classes | `experiments/public_history_awards.py`, `common_budget_global.py` | `results/public_history_awards.json`, `common_budget_global.json` |

Filenames without a repeated directory prefix in the table are in `experiments/`.
Table layout remains in the article's LaTeX source; the saved results give its
numerical entries.

## Applied planning comparison

```sh
python experiments/stanford_award_study.py
python experiments/stanford_case_sensitivity.py
python experiments/plot_stanford_awards.py
```

The financial mapping protects $15,000 for each of two candidate finalists and
$10,000 for third place, leaving $35,000. It retains the published ceiling of
two $2,500 sharing awards. This is a proposed two-team planning benchmark, not
a structural fit of all entrants and winners in the actual competition.

The case joins final-ranked nonbenchmark teams to predeadline-flagged submissions
and the originating notebook version. A capability count requires that the final
team leader authored the source version before April 23 and that it later met
an explicitly chosen private-validation TM threshold. The primary threshold is
0.3; 0.2 and 0.4 are sensitivity diagnostics, not the official sharing criterion.
SubmissionDate is a September re-evaluation date and is not used as the code's
creation clock. Coverage and unmatched records are reported. Missing metadata
are not treated as evidence of non-readiness.

The documented fraction supplies a readiness floor only under the explicit
planning translation stated in the paper. Technology and recognition ranges
remain sensitivity assumptions. `stanford_case_sensitivity.py` changes the
rediscovery lower bound through 0.001, 0.01, 0.05, and 0.1 while retaining the
published $2,500 payment. Each row has 375 scenarios and one common share rounded
upward by 0.1 percentage point. Its no-bonus comparator meets the same uniform
release requirement. The driving scenario and the resulting winning stakes are
saved explicitly. A 0.409 ratio at the most extreme bound is not a typical gain.

The paired comparison instead tailors shares to known parameters. At exact
release thresholds, 171 of 286 positive-threshold settings accelerate under
reservation, with median time ratio 0.967; all 286 slow under recycling at this
payment. These are boundary limits of strict-release contracts. Rounding can
change the sign in small cases: the primary rounded comparison includes seven
slight recycling accelerations, which remain in `paired_settings.csv`. The case
therefore makes a directional claim for the common-contract exercises and the
specified payment, not a universal no-recycling theorem.

Proposed pool-screening floors 0.5 and 0.8 reduce reservation's common share to
50.0% and 15.3% and raise its time ratios to 0.894 and 0.970. These are hypothetical
information scenarios, not measured readiness rates. All bound, comparator,
rounding, and screening outputs are in `results/stanford_case_sensitivity/`.
The general contest solver independently checks 11,403 continuation times and
strict-release conditions; threshold boundaries are also checked, with positive
perturbations. `stanford_award_study.py` preserves auxiliary capability-threshold
and $50-step bonus-menu calculations, including the original 7,500 release checks.

`data/stanford_rna/README.md` explains the linkage, coverage, missingness, license,
and source clocks. To reacquire the pinned public snapshot, optionally run:

```sh
python experiments/fetch_stanford_activity.py --refresh
```

This streams approximately 8.7 GB of upstream CSV data, retaining only small case
subsets. No Kaggle credentials are required. Normally no download is needed.
Snapshot URLs, upstream and extracted-file hashes, retrieval times, and the data
license are included. The provider explicitly filters the public database.

## Other model calculations and checks

```sh
python experiments/reproduce.py
python experiments/budget_rule_comparison.py
python experiments/common_budget_global.py
python experiments/recycling_profile.py
python experiments/budget_region_map.py
python experiments/discounted_budget_robustness.py
python experiments/contract_extensions.py
python experiments/accounting_strength.py
python experiments/public_history_awards.py
python experiments/institutional_contracts.py
python audit/code/accounting_strength_independent.py
python audit/code/public_history_awards_independent.py
```

- `common_budget_global.py` computes attainable stakes and reservation welfare.
- `recycling_profile.py` solves the concave bonus problem and numerically searches
  the outer share dimension; the outer search is not a general global certificate.
- `budget_region_map.py` evaluates 59,049 local slope settings; `--reuse-grid`
  redraws saved data without recomputation.
- `discounted_budget_robustness.py` checks six timing cases and 5,043 matched
  discounted settings; its additional figure is preserved as an auxiliary result.
- `accounting_strength.py` records 207 limiting parameter checks and 19,683 speed
  magnitudes on matched 81-by-81 technology grids. Distribution summaries describe
  equally weighted grid cells, not empirical population probabilities.
- `public_history_awards.py` computes the broader winning-award benchmarks and a
  baseline-specific rediscovery boundary. Certification remains available for play.
- `institutional_contracts.py` reconciles the announced 2025/2026 award schedules;
  these schedules are not actual disbursement records or causal policy treatments.

Auxiliary release-region, discounted-region, curvature, and sensitivity graphics
remain available in `figures/` and their generating scripts. The current article
uses only the three figures in the index. The repository is standalone and requires no external working-project folders.

## Data license

The included Meta Kaggle subsets retain their Apache 2.0 license in
`data/stanford_rna/LICENSE_DATA.txt`. Source URLs, snapshot identifiers, and
checksums are recorded alongside the data. This dataset license does not grant
a separate license for the analysis code.
