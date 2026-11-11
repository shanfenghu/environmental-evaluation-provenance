# Selected assessment outputs

These are retained scientific outputs from the recorded assessments. They are
provided as files alongside their generation code. No provider observations,
prepared predictor/response datasets or fitted model weights are included here.
The four tasks combine EPA/CAMELS-GB with dispersed/northern-edge cohorts.
EPA errors use ppb; hydrological errors use mm/day.

## Figures and readable tables

- [Operation](../../figures/source-assessment/operation.pdf) is a constructed
  explanatory example, not an additional empirical dataset.
- [Direct contrasts](../../figures/source-assessment/direct.pdf) show matched
  exclusion sensitivity across the original confirmation cycles.
- [Order allocation](../../figures/source-assessment/order.pdf) shows forward,
  reverse and averaged training allocations with descriptive uncertainty.
- [Confirmation](../../figures/source-assessment/confirmation.pdf) retains the
  predicted-minus-measured contrasts, including the failed EPA forecasts.
- [External errors](../../tables/source-assessment/external.csv) retain all three
  learners' full-fit external MAEs.
- [Raw exclusion changes](../../tables/source-assessment/exclusion-means.csv) retain
  each learner's own-minus-other exclusion MAE, averaged over the original four
  confirmation cycles per task. These are changes on matched development sites,
  not external errors or the twelve-bank follow-up. Subtracting learner columns
  gives the relative confirmation contrasts.
- [Allocations](../../tables/source-assessment/allocation.csv) retain coupling,
  refit, cohort and total-gap components for the trees-minus-ridge comparison.
- [Overlap](../../tables/source-assessment/overlap.csv) distinguishes the fixed-model
  decomposition from the exclusion comparison and labels unsupported edge cases.
- [Precision and cost](../../tables/source-assessment/precision-cost.csv) retain
  the primary trees-minus-ridge timing-adjusted comparison, using joint
  three-learner timing costs.
- [Source profiles](../../tables/source-assessment/source-profiles.csv) retain
  heterogeneous source contrasts, their quantiles and trimmed summaries.

Six authored reference CSVs also preserve the method displays:
[comparison targets](../../tables/source-assessment/comparison-targets.csv),
[notation](../../tables/source-assessment/notation.csv),
[query roles](../../tables/source-assessment/query-roles.csv),
[cycle operations](../../tables/source-assessment/cycle-operations.csv),
[domain counts](../../tables/source-assessment/domain-counts.csv) and the
[fixed-rule example](../../tables/source-assessment/fixed-rule-example.csv).
The fixed-rule example is constructed, not provider data. Query-role external
entries require nonempty scored cells; the direct contrast does not use them.
The reference CSVs are authored sources; domain counts agree with the recorded
source assignments in both tasks per domain.

The PDF files are the selected vector figures. CSV decimals are display rounding;
full numerical precision and qualifications are retained in the JSON results.
The PDF-embedded DejaVu fonts have their own [licence](../../docs/embedded-font-license.txt).
Provider and other required credit remain in [the notices](../../THIRD_PARTY_NOTICES.md).

## Numerical results

- [summary.json](summary.json): original joint summaries, covariance/uncertainty,
  confirmation comparisons, external errors and secondary allocations.
- [precision.json](precision.json): 48-bank conditional averaging, descriptive
  whole-bank bootstrap ranges and timings for six endpoints. The primary
  trees-minus-ridge contrast has no clear timing-adjusted benefit; several
  secondary ridge endpoints have descriptive gains using the same joint cost.
- [orders/analysis.json](orders/analysis.json): conditional order enumeration,
  cyclic comparisons and order sensitivity.
- [profiles/run.json](profiles/run.json) and its four case files: source-level
  contrasts, observed coverage, heterogeneous signs and sensitivity summaries.
- [rankings/run.json](rankings/run.json) and its four case files: learner rankings,
  changed accuracy gaps and recorded reversal checks.
- [cost/run.json](cost/run.json): retained timing-probe summaries and sensitivity
  diagnostics; these are not new timing measurements.
- [display-data.json](display-data.json) and [illustration.json](illustration.json):
  numerical plotting points and the constructed example's arithmetic.

The outcome losses and original reference/follow-up records remain under
`records/`; these are derived assessment results, not provider input copies.
The selected results retain failed forecasts, weak/mixed ridge contrasts,
unchanged learner rankings and uncertain cost/precision benefits. They do not
establish causal effects, coverage guarantees or independent replication.

## Generation route

Run the seven recorded-result commands in the [root README](../../README.md)
into a new `outputs/` tree. They require no provider input download or model fit.
The generated file mapping is:

- `outputs/display/figures/*.pdf` to `figures/source-assessment/`.
- The six numerical `outputs/display/tables/*.csv` files to `tables/source-assessment/`.
  The six authored reference CSVs are retained directly, without a numerical generator.
- `outputs/summary.json`, `precision.json`, `orders/`, `profiles/`, `rankings/` and
  `cost/` to the corresponding paths in `results/source-assessment/`.
- `outputs/display/display-data.json` and `illustration.json` to this directory.

Source assignments, seeds, weights and bootstrap settings are retained in the
records/configuration. Compare numerical summaries at absolute tolerance 1e-8,
which is below displayed result precision, and inspect any discrepancy. PDF
creation timestamps and analysis elapsed times can change without changing a
scientific display or finding. Regeneration must not overwrite these selected
files or silently replace a historical result. New provider-input/native fitting
runs use the separate [data recipe](../../data/README.md).
