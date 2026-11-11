# Source-aware retrospective assessment

A later prediction score depends on both the fitted sources and the scored
sources. This module compares two fits on exactly the same later sites: one fit
excludes their earlier records; the other excludes an equally sized disjoint
set. Site counts match within every geographic cell. Daily-row counts,
geographic coverage and source histories can still differ. This is an
algorithmic source-availability comparison, not an isolated environmental
causal effect or a new validation primitive.

`queries.FitScores` binds a source-indexed loss matrix to its input version,
recipe version, fitted sources and scored sources. The loss columns are ridge,
histogram boosting and a seasonal reference. `ScoreQuery` additionally declares
temporal-reference, entity-held-out or external-held-out roles. A state may
support different roles on different scored sets. Chronology alone does not
establish entity exclusion. The guards check declared local execution records,
not unknown provider lineage or independent environmental measurements. Recipe
keys hash the declared learner configuration and the seasonal/response-transform
labels. They do not hash implementation bytes, package versions or the boosting
seed fixed in `models.py`. Query guards compare the declared key and source/input
fields; they do not automatically detect changes in those omitted software details.

The enclosing records document selected configuration/environment context. The
retained original run has an empty `source_sha256` field, and the retained pairing
run has no implementation source-hash mapping. The supplied records therefore do
not verify the historical implementation-to-run association for those fitted
losses. Later analysis hashes identify that analysis, not all earlier fitting.
Current replay checks numerical consistency; it cannot recover that missing
historical binding. Historical fields remain unchanged. These keys are not portable
cross-version cache certificates.

Each source loss averages retained days, then sources within a cell and cells
equally. Joint learner differences are formed from the same raw losses. For
validation sources V and disjoint, equal-count W, the direct cell audit is
`S_c = score(V_c, fit_without_V) - score(V_c, fit_without_W)`.
Here `U_c` is the eligible development set in cell `c`, excluding reserved
external sites and the protected fitted core. The disjoint blocks `V_c` and
`W_c` are drawn from `U_c`, each with `q_c` sites; remaining eligible sites may
also contribute to fitting. The held-out/eligible reference is
`C_c = score(V_c, fit_without_V) - score(U_c, fit_without_V)`.
Under the uniform symmetric disjoint-pair law,
`E(S_c) = E(C_c)/(1-q_c/|U_c|)`.
This is an assignment expectation on retained data, not equality in every
realised fold or a guarantee for a small empirical forecast.

The cohort file fixes the inherited dispersed external cohort using deterministic
within-cell hash selection. It is not a probability sample or a distance-balanced
design. Northern-edge
cohorts are the northernmost fifth within four declared cells. Two fixed-hash
sources per cell form a protected fitted core. Other sources are partitioned
into five disjoint quota blocks using fresh recorded cycle seeds; neighbouring
blocks supply W. Eight original cycles per task are retained. Cycles, rather
than dependent folds or daily observations, are the replication units.

`models.fit_native` refits a StandardScaler and Cholesky ridge (penalty 100) on
exactly the selected earlier sources. Histogram boosting uses absolute-error
loss, 100 iterations, learning rate 0.1, 31 leaves, minimum leaf size 20,
regularisation 0.1, no early stopping and random state 41. The seasonal reference
is a raw earlier-response median by cell/month, fitted on the same sources.
Hydrological responses use log1p for fitting; predictions use expm1 after
clipping to [-20,20] and are bounded below by zero. Losses use original units.
These inherited settings define a reproducible comparison, not tuned optimality.

`native.disde` uses five-fold cross-fitted standardised logistic domain
classification (regularisation parameter 1, max iterations 1000), earlier
predictor means and probabilities clipped to [1e-6,1-1e-6]. Its cohort labels and
covariates are allowed diagnostic information; external responses never enter
predictive fitting. Normalised overlap-weight parts sum to a fixed-model
cohort contrast. They do not allocate the changing-fit gap or create shared
support. Both edge tasks have disjoint observed latitude ranges in every cell.

The crossed baseline reports both fitting/cohort orders, their two-factor
Shapley average and their interaction. The telescoping membership/refit/cohort
path is a descriptive reference convention, not a unique mechanism allocation.
The summary code resamples whole cycles jointly with 3,999 bootstrap replicates.
Eight-cycle joint bands and independent four-cycle-half intervals are descriptive
small-sample approximations, not exact or population coverage statements.

The reference records preserve two failed EPA finite-half forecasts, weak ridge
contrasts and all four tasks. Trees have positive direct relative contrasts in
every confirmation cycle but also the lowest observed full-fit external MAE.
A larger relative exclusion contrast does not imply poorer absolute prediction.
The observations were inspected before this assessment; repeated assignments do
not add datasets or fresh environmental populations. No runtime superiority,
causal mechanism, decision benefit or precise forecasting guarantee is asserted.

## Balanced block pairing

`source_assessment.pairing` operates on a finite `k × k × d` loss matrix:
row `i` is the later scored block and column `j` the excluded earlier block.
`paired_audit` uses one cyclic neighbour per block. `averaged_audit` averages all
other fits; `interaction_matrix` removes additive row and column effects.

Both comparisons hold the scored block fixed and balance fitted-state weights.
The average is the conditional mean over uniform global cyclic block orders,
so total covariance gives no larger unconditional covariance under the declared
exchangeable-block design. The guarantee does not require an environmental model
or independent source losses. It assumes the fitted rule attached to a source
set does not change when block indices are renamed. Fixed learner randomness
satisfies that condition. Outcome-selected neighbours do not obey this law.

Averaging removes order variation, not block-selection or population uncertainty.
It need not reduce an observed small-bank sample SD. Extra block predictions
cost work if source-indexed losses were not already retained. Additive cohort
and overall fitted-state effects cancel, but interactions involving geometry,
row counts and source history remain. No causal attribution or runtime advantage
is implied.

From the repository root, run this API example with the source path set explicitly.
The experiment launchers set that path themselves; no package installation is needed.

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import numpy as np
from source_assessment.pairing import averaged_audit
# Fixed scored-block difficulty plus an own-exclusion increment of two.
g = np.arange(5)[:, None] + np.zeros((5, 5)) + 2 * np.eye(5)
assert np.isclose(averaged_audit(g), 2)
PY
```
