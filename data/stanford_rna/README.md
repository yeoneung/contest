Stanford RNA case records
========================

Source: Kaggle, Meta Kaggle, version 2401, updated November 20, 2026.
https://www.kaggle.com/datasets/kaggle/meta-kaggle
License: Apache 2.0. The data provider warns that these are filtered and
transformed public records, not a complete database dump.

The files retain two competition identifiers: 87793 (2025) and 118765 (2026).
The article's applied analysis uses 2025; the second edition documents the
comparison of announced award schedules at the same total budget.

From the repository root:

    python experiments/stanford_award_study.py

This uses the supplied small case subsets. Reacquiring them is optional:

    python experiments/fetch_stanford_activity.py --refresh

The fetch streams approximately 8.7 GB of upstream CSV data without storing
those large tables. It does not require a Kaggle account or API credentials.
Snapshot-specific source and extracted-file hashes appear in the provenance
JSON files. If remote APIs or version retention change, use the included
hash-verified subsets. No notebook source code, participant names, or emails
are redistributed. Pseudonymous team-leader and notebook-author identifiers
are retained only to verify ownership of a dated version.

Linkage and clocks
------------------
Teams.CompetitionId selects the case. Submissions.TeamId links submitted
outputs. SourceKernelVersionId links the immutable notebook version, whose
AuthorUserId must match the final TeamLeaderId for the capability count.
CreationDate, not SubmissionDate, is the capability clock: the 2025 selected
entries were re-evaluated in September, after the May submission deadline.
The final private score supplies a retrospective quality label. It is not a
historical leaderboard observation. Qualification thresholds 0.2, 0.3, and 0.4
are analytical diagnostics, not the official VFOLD early-sharing criterion.

The primary retrospective pool consists of final-ranked nonbenchmark teams.
An owned predictor created before April 23 and later passing validation
establishes an existing capability for its author. Interpreting it as a lower
bound on the model's method readiness requires the explicit planning mapping
and independent candidate draws stated in the paper. Unmatched versions,
non-leader authorship, and later selected versions remain unclassified; they
are not coded as non-ready. The snapshot does not identify research-cost
coefficients or the effect of rewards on discovery time.

Financial mapping
-----------------
The announced 2025 awards are $45,000, $15,000, $10,000, plus two $2,500 sharing
awards. For a two-finalist planning contract we protect $15,000 per finalist
and $10,000 for third place. The remaining $35,000 covers the incremental
winning award and up to two sharing payments. Protected placements and the
retained bonus ceiling are restrictions on this proposed planning comparison,
not a claim that the entire actual contest was a two-team race. The full
$70,000 ranked-prize pool is not substituted for a single winning stake.
Recycling unearned sharing funds is a proposed alternative, not an identified
historical treatment in the Stanford contest.

Observed amounts and dates, derived capability bounds, and unobserved
technology/recognition scenarios are distinguished in the manuscript's input
table. The primary case comparison uses zero researcher discounting; this
also makes protected placement amounts irrelevant to marginal effort.
