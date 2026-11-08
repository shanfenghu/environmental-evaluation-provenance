# Reproduction inputs

`prepared/` contains metadata for two exact numeric archives; obtain or reconstruct
the arrays before native fitting or role verification. Keep these files outside
Git. The small source-indexed outcome losses under `records/` suffice for recorded
aggregate, covariance, follow-up and display reconstruction.

- `air_quality_2021_2022.npz`: SHA-256 `d79bad83190d95c5bb4140e93407852cb82aa5b86de3064d6f9e1b00fab1fdc6`.
- `hydrology.npz`: SHA-256 `767e1d2267da13e288cf8f73382135a45d2a7d5195c4016fa7a1af5337a5c934`.

The metadata record source identities, dates, coordinates, cells and row counts.
The loader verifies exact array bytes. Copy exact supplied arrays to
`data/prepared/`, or select an explicit directory with `--data` for native commands.
No public download location for these frozen prepared assets is asserted here.

To reconstruct from provider data, use the three declared archives in
`configs/acquisition-v1.json`: daily EPA NO2 summaries for 2021 and 2022 and the
original CAMELS-GB 2020 data release. URLs, original sizes/hashes and every used
scientific member's size/hash are pinned there. Existing files are verified,
never replaced. Acquisition requires network access and upstream availability:

```sh
.venv/bin/python experiments/run_source_assessment.py acquire --raw data/raw
.venv/bin/python experiments/run_source_assessment.py prepare --raw data/raw --output outputs/prepared
```

Preparation checks containers and scientific member bytes, applies
`configs/preparation-v1.json`, and writes arrays and metadata to a new directory.
Compare array names, dtypes, shapes and values against the exact frozen arrays.
A regenerated NPZ container may have different bytes despite identical arrays;
its SHA and query identifiers then differ. Metadata storage fields may also
differ. Do not overwrite frozen arrays or call a repack byte-identical recovery.

CAMELS wrappers can change when downloaded again. Strict mode rejects a changed
container. Only explicit `--allow-container-repack` accepts a different wrapper
when *all used scientific members* still match. A changed scientific member
always fails. Retain actual provider bytes separately: hashes and mutable URLs
are not backups. Fresh upstream acquisition is not part of an offline replay.

See [input definitions and provider boundaries](../docs/source-assessment-inputs.md)
and [required attribution](../THIRD_PARTY_NOTICES.md). EPA data are public domain;
CAMELS-GB data use Open Government Licence v3. No provider endorsement is implied.
Contemporary hydrological forcings support retrospective assessment. The recorded
boundary does not certify unknown upstream measurement ancestry or independent
sampling of environmental populations.
