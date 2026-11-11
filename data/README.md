# Reproduction inputs

Download the public inputs from their original providers. This repository does
not distribute provider observations or prepared copies, through Git or release
assets. Keep downloaded archives and locally prepared arrays outside Git.
Our small derived outcome records in `records/` support analysis without a new
model fit; they are scientific results, not replacement input observations.

## Providers and versions

EPA AQS/AirData daily nitrogen-dioxide summaries (parameter 42602), calendar years
2021 and 2022, are public domain. See the [pre-generated file catalogue](https://aqs.epa.gov/aqsweb/airdata/download_files.html),
[file definitions](https://aqs.epa.gov/aqsweb/airdata/FileFormats.html) and
[EPA permission statement](https://www.epa.gov/outdoor-air-quality-data/do-i-need-request-permission-use-monitoring-data-and-graphics-airdata).
The declared downloads are [2021](https://aqs.epa.gov/aqsweb/airdata/daily_42602_2021.zip)
and [2022](https://aqs.epa.gov/aqsweb/airdata/daily_42602_2022.zip), saved as
`epa_no2_2021.zip` and `epa_no2_2022.zip`.

Use the original CAMELS-GB 2020 dataset, not a later expanded release:
Coxon et al. (2020), *Catchment attributes and hydro-meteorological timeseries for
671 catchments across Great Britain*, NERC Environmental Information Data Centre,
[DOI and citation](https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9).
The [provider catalogue](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9)
has its Download/Access section and [whole-dataset ZIP](https://data-package.ceh.ac.uk/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip),
saved as `camels_gb_2020.zip`. Access constitutes agreement to the provider's
[Open Government Licence terms](https://eidc.ac.uk/licences/ogl/plain) and citation
requirements. Preserve UKCEH, University of Bristol, Environment Agency and BGS
credit in [the notices](../THIRD_PARTY_NOTICES.md). The dataset description article's
CC BY licence is distinct from the data licence. No provider endorsement is implied.

## Selection and preprocessing

`configs/preparation-v1.json` defines the transformations; the implementation is
`src/source_assessment/prepare.py`. EPA uses the continental bounds 25–50 degrees
latitude and -125 to -66 longitude, at least 18 observations per day and 300 valid
days in each year. It chooses one POC (Parameter Occurrence Code) per site by earlier-year
qualifying record count, breaking ties by code. It rejects the selected site/POC series if dates repeat or
coordinates change; it does not deduplicate rows. The recorded version retains
353 sites. Its response is the provider's daily `Arithmetic Mean`, in ppb. Fitting uses 2021; scoring uses 2022.

CAMELS-GB uses daily specific discharge (`discharge_spec`), catchment-area-normalised
flow in mm/day, and declared meteorological/static
features, fitting 1995-10-01 through 2005-09-30 and scoring 2005-10-01 through
2015-09-30. At least 90% valid rows are required in each period; negative flow and
nonfinite predictors/responses are excluded. The recorded selection has 618
catchments. Seasonal/coordinate features and rolling precipitation transforms
are fixed before fitting. Hydrological fitting responses use log1p, with clipped
inverse predictions. Earlier responses supply pooled training examples, not explicit
site-specific lagged-response predictors. Contemporary forcings support retrospective
assessment.
See [input definitions](../docs/source-assessment-inputs.md) for the feature and
source-role boundaries. Geographic cells and scoring weights are fixed by the
configuration, not inferred from model performance.

The four cells split EPA coordinates at latitude 39 and longitude -100, and
CAMELS-GB at latitude 53.5 and longitude -2.5. Equality goes north/east.
For each cell, the dispersed external cohort is the first `floor(n/5)` site IDs
in ascending `sha256(f"{seed}:{site_id}".encode()).hexdigest()` order, with seed
301 for EPA and 41 for CAMELS-GB. This reconstructs the retained cohort lists
exactly. It is a fixed inherited identity rule, not distance balancing or a
probability sample of environmental sites. The northern-edge cohort instead
takes the northernmost `floor(n/5)` IDs per cell, breaking latitude ties by ID.
These fixed cohort selections are distinct from the later random cycle
assignments of eligible development sites. Each cell receives scoring weight 1/4.

## Download and prepare

After the Python setup in the root README, run from the repository root:

```sh
.venv/bin/python experiments/run_source_assessment.py acquire --raw data/raw --allow-container-repack
.venv/bin/python experiments/run_source_assessment.py prepare --raw data/raw --output outputs/prepared --allow-container-repack
```

Alternatively, download the three provider ZIPs above in a browser and save them
under those filenames in `data/raw/`, then run `prepare`. The output directory
must be new. Preparation reads declared scientific CSV members directly from the
ZIPs and writes two NPZ archives, matching JSON metadata and a verification record.
Every newly written NPZ is hashed in its own metadata; the native loader checks
that pairing. Select this generated directory with `--data outputs/prepared`.
No separately supplied frozen NPZ or metadata download is needed.

`configs/acquisition-v1.json` records provider URLs, reference container identities
and the scientific member hashes used in the recorded analysis. Provider ZIP
wrappers can change. The explicit repack option accepts a different container
only when all used scientific members still match; strict mode omits the option.
A changed used member is rejected, rather than silently analysed as the old
version. If that happens, retain the new bytes and investigate the provider change;
do not disable validation or imply that the historical version was recovered.
Container hashes are provenance checks, not a demand for byte-identical generated
NPZs or model outputs. Existing downloads and prepared inputs are never overwritten.

Allow about 255 MiB for compressed raw downloads, about 120 MiB for prepared
arrays, at least 2 GiB free temporary disk and several GiB RAM. The lock targets
Python 3.12.14. A full native run is much slower than preparation or record analysis;
its elapsed time depends on hardware. Provider services are external dependencies:
links/catalogue access do not establish that a fresh complete download succeeded.
Retain permitted local input versions separately for long-term reproducibility.

## Run and compare

The root README lists commands reconstructing all recorded aggregates, follow-up
summaries and figures/tables without input downloads or model fitting. To check
native fitting practically, start with the four full fitted states:

```sh
.venv/bin/python experiments/run_source_assessment.py fit --data outputs/prepared --full-only --output outputs/full-states
.venv/bin/python experiments/run_source_assessment.py compare --replay outputs/full-states --scientific --atol 1e-8 --output outputs/full-states-comparison.json
```

Scientific comparison checks all replayed losses and scientific records, including
source memberships, counts, recipes and allocations. It excludes elapsed timings,
metadata fingerprints and input/query digests that legitimately change when inputs
are regenerated. The output records both input versions. Absolute tolerance 1e-8
is far below the reported result precision in either response unit; a failure
requires investigation, not automatic widening. A successful `compare` exits with
status zero and writes `complete: true`. For the documented `--full-only` check,
inspect `stage: "full_only"`, `cases: 4`, `native_source_states_replayed: 4` and
`full_native_run_replayed: false`: the check covers four full fits, not the
whole exclusion experiment. `maximum_absolute_numeric_difference` must not exceed
`absolute_tolerance`. A failed comparison raises an error and exits nonzero.
Different software or provider
versions may produce larger changes. These checks establish neither statistical
coverage nor new-population replication. Omit `--scientific` for the optional exact
input-version check; `verify` likewise audits the historical fingerprinted losses
and is not required for newly generated input containers.

For all original fits, omit `--full-only` and use a new output directory. The full
recipe has 164 source states (328 candidate and 164 seasonal fits). Seeds and
source assignments are fixed. To rerun the 48-bank follow-up from regenerated
inputs, first complete that original native run as `outputs/native`, then direct
its reference configuration to those newly generated records:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('configs/pairing-v1.json').read_text())
cfg['reference_records'] = 'outputs/native'
Path('outputs/pairing-config.json').write_text(json.dumps(cfg, indent=2) + '\n')
PY
.venv/bin/python experiments/run_pairing_precision.py --config outputs/pairing-config.json --data outputs/prepared --output outputs/pairing-native
.venv/bin/python experiments/analyze_pairing_precision.py --records outputs/pairing-native --output outputs/pairing-native-analysis.json
.venv/bin/python experiments/check_pairing_cost_sensitivity.py --records outputs/pairing-native --data outputs/prepared --output outputs/cost-native
```

The follow-up fits 240 states and repeats resident queries; fresh timings will not
match recorded timings. Judge substantive contrasts, signs, rankings and uncertainty
alongside numerical agreement. Preserve failed forecasts, weak/mixed ridge contrasts,
source heterogeneity and uncertain precision–cost benefit. Neither matching outputs
nor completeness filters certify unknown upstream measurement ancestry or independent
sampling of environmental populations.
