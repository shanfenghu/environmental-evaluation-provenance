# Source-aware environmental assessment

Compare later prediction errors when different sites' earlier records are left
out of model fitting. Both fits predict the same later observations: one leaves
out the scored sites, and the other leaves out equally many different sites in
each geographic region. The software fits learners, records site-level losses
and reconstructs the resulting comparisons. Read [the methods](docs/source-assessment.md) and
[the input guide](data/README.md) before interpreting results.

Use Python 3.12.14 and `uv`. From this directory:

```sh
uv venv --python 3.12.14 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements.lock
.venv/bin/python experiments/test_source_assessment.py
```

If `uv` cannot discover an installed interpreter by version, supply its executable
path explicitly: `uv venv --python /path/to/python3.12 .venv`. The executable must
report Python 3.12.14; a managed interpreter download is not required.

The launcher imports only this directory's `src/`; installation is unnecessary.
The lock pins all numerical and plotting dependencies. Original authored software,
tests, configuration and documentation use the [MIT licence](LICENSE). Provider
data, prepared metadata, derived records, dependencies and embedded fonts retain
their separately applicable terms; see [the attribution](THIRD_PARTY_NOTICES.md).

For a first result, open [the exclusion-change table](tables/source-assessment/exclusion-means.csv).
Its four rows report original confirmation means by task. Subtract `ridge` from
`trees`: the extra exclusion-related rise is 1.10–1.26 ppb for air quality and
0.0626–0.0644 mm/day for flow (rounded). This concerns matched development sites;
the separate external-error table measures prediction at reserved sites. To
reconstruct these rows, run the recorded-result commands below and inspect
`outputs/display/tables/exclusion-means.csv`.

The actual selected [figure PDFs](figures/source-assessment/),
[readable CSV tables](tables/source-assessment/) and
[key numerical results](results/source-assessment/) are included. The
[selected-output guide](results/source-assessment/README.md) explains each file,
its interpretation and its generation route. These scientific outputs remain
available alongside the code and derived outcome records.

Small reference records and source-indexed losses are included. Prepared arrays
and provider archives are separate inputs, excluded from Git. Without prepared
arrays you can reconstruct the selected numerical summaries and displays:

```sh
mkdir -p outputs
.venv/bin/python experiments/run_source_assessment.py summarize --output outputs/summary.json
.venv/bin/python experiments/analyze_fold_pairing.py --output outputs/orders
.venv/bin/python experiments/analyze_pairing_precision.py --output outputs/precision.json
.venv/bin/python experiments/analyze_source_inclusion.py --output outputs/profiles
.venv/bin/python experiments/analyze_assessment_rankings.py --output outputs/rankings
.venv/bin/python experiments/check_pairing_cost_sensitivity.py --recorded records/cost/run.json --output outputs/cost
.venv/bin/python experiments/run_source_assessment.py render --pairing outputs/precision.json --profiles outputs/profiles --output outputs/display
```

These commands reconstruct parallel analyses rather than one chained pipeline.
`render` reads the retained original records and their summary in
`records/reference/`; it consumes the regenerated pairing and profile results
through the two explicit options. It does not read `outputs/summary.json`,
`orders/`, `rankings/` or `cost/`. Those are separate reconstructions for numerical
comparison with the selected results; the rendered key table uses the retained
original summary.

Each destination passed through `--output` must be absent; its shared parent,
such as `outputs/`, may already exist. These commands use the frozen original and
12 follow-up banks per task (48 across four tasks). Each original task has eight
cycles: four for the forecast and four for its direct confirmation. A cycle or
follow-up bank contains five dependent blocks; those blocks and their days are
not separate replicates. The forecast uses the first four cycles' held-out-minus-
all-eligible-site scores, divides each cell's contrast by its retained fraction,
then averages cells and cycles. Its measurement is the direct own-minus-other
audit from cycles 5–8 on the same environmental records. `confirmation.pdf`
plots forecast minus measurement. The precision bootstrap resamples whole banks jointly;
source profiles report observed coverage without filling missing sources.
Recorded timing reconstruction performs no fresh timing measurements. The display
command writes four PDFs and six neutral CSV tables, plus arithmetic/point data.
Allocation and overlap CSVs support secondary numerical comparisons. JSON retains
full precision; displayed decimals are rounded only at export.

Download provider inputs and preprocess them using [the data recipe](data/README.md).
No provider input archives or prepared copies are distributed through this repository
or release assets. For a practical native check, run the four full fitted states:

```sh
.venv/bin/python experiments/run_source_assessment.py fit --data outputs/prepared --full-only --output outputs/full-states
.venv/bin/python experiments/run_source_assessment.py compare --replay outputs/full-states --scientific --atol 1e-8 --output outputs/full-states-comparison.json
```

The comparison retains numerical and source-membership checks while allowing
regenerated container/query fingerprints to differ. The data README explains the
comparison tolerance, optional exact-version checks and provider access limits.

For the complete native recipe, omit `--full-only` (164 source states: 328 native
candidate fits and 164 seasonal fits). `--fixed-cycle` instead replays 24 source
states from the first original cycle of each task. Native launchers fix BLAS and
OpenMP to one thread before numerical imports. Arrays occupy approximately
120 MiB compressed; raw archives approximately 255 MiB. Allow several GiB RAM
and at least 2 GiB temporary disk. Whole native runs are substantially slower than
record reconstruction; their timings depend on hardware and system load.

```sh
.venv/bin/python experiments/run_source_assessment.py fit --data outputs/prepared --output outputs/native
.venv/bin/python experiments/run_source_assessment.py compare --replay outputs/native --scientific --atol 1e-8 --output outputs/native-comparison.json
.venv/bin/python experiments/run_source_assessment.py summarize --records outputs/native --output outputs/native/summary.json
PYTHONPATH=src .venv/bin/python - <<'PYTHON'
from source_assessment.tables import render_tables
render_tables('outputs/native', 'outputs/native-tables')
PYTHON
```

Read `outputs/native-tables/exclusion-means.csv` for the key table from the
newly fitted records. This needs the complete original eight-cycle run;
`--full-only` does not supply the exclusion cycles. The table API writes the
original-study tables without requiring the separate twelve-bank follow-up.

The data README also gives the follow-up configuration using those regenerated
native records, rather than requiring historical input byte fingerprints.

The follow-up executor uses the unchanged 48 recorded seeds and fits 240 source
states. It predicts two blocks versus all five using the same fitted learners,
charges the shared fit to both plans and alternates execution order. The cost
probe refits four fixed states and performs ten paired queries per state, dropping
two warm-ups. Fresh elapsed times will not equal the retained timings. These
routes repeat the same scientific settings; they supply no new environmental
population, independent replication or deployment evidence.

The reference records retain failed finite-half forecasts, weak/mixed ridge
contrasts, unchanged learner rankings, source heterogeneity, observed SD increases
and endpoint-dependent timing-adjusted results: the primary trees-minus-ridge
contrast has no clear benefit from extra scoring, while several secondary ridge
endpoints have descriptive gains under the same joint three-learner cost.
Lower external prediction error can coexist with greater exclusion sensitivity.
Matching site counts does not match daily
row counts, geography or histories. Conditional averaging removes fold-order
variation; it does not remove bank-selection or population uncertainty. Crossed
allocations depend on the chosen path, and overlap diagnostics have a different
fixed-model target and require support. No causal or coverage guarantee is implied.
