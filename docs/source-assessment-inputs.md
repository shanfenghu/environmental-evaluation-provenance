# Inputs and attribution

Local preprocessing creates the two numeric archives used by the
native recipe: `air_quality_2021_2022.npz` and `hydrology.npz`. Their source-indexed
arrays contain earlier predictors/responses (`x`, `y`), later predictors/responses
(`z`, `w`) and date indices (`t`, `s`). Source order, identities, coordinates,
cell assignments and row counts are retained in the matching metadata. The
first prepared feature is constant and is dropped by the native loader; the
resulting native predictor dimensions are 13 and 20. Only hydrological fitting
responses transform to log1p. Record provider and local preprocessing versions when generating new inputs.

EPA nitrogen-dioxide summaries retain 353 sites, fitting in 2021 and scoring in
2022. Geographic bounds are 25–50 degrees latitude and -125 to -66 longitude;
days require at least 18 observations. A single POC is chosen by earlier-year
completeness, with deterministic tie-breaking. Ambiguous site dates and coordinate
drift are excluded; each year needs at least 300 retained valid days. Negative
reported means are retained. Geographic cuts are latitude 39 and longitude -100.

The 13 EPA predictors are fixed coordinate and seasonal terms. With
`u = (latitude - 40)/10`, `v = (longitude + 20)/50` and
`a = 2*pi*day_of_year/365.25`, they are `u`, `v`, `u*u`, `v*v`, `u*v`,
`sin(a)`, `cos(a)`, `sin(2*a)`, `cos(2*a)`, `u*sin(a)`, `u*cos(a)`,
`v*sin(a)` and `v*cos(a)`. The constant column produced during preparation
is removed before fitting. These predictors contain no earlier response values.

The original CAMELS-GB 2020 release supplies 618 retained catchments. Fitting is
1995-10-01–2005-09-30 and scoring 2005-10-01–2015-09-30, with at least 90% valid
days in each period. Negative flow and nonfinite response/predictor rows are
invalid. Geographic cuts are latitude 53.5 and longitude -2.5. Fixed coordinate
and seasonal terms are supplemented by transformed precipitation, seven-/thirty-
day precipitation means, temperature, potential evapotranspiration, area and
elevation. Contemporary observed forcings make this retrospective assessment;
they are not an operational forecasting promise. Completeness filters do not
establish representative environmental sampling.

Hydrology uses the same 13 coordinate/seasonal terms plus seven features:
`log1p(P)/3`, `log1p(mean7(P))/3`, `log1p(mean30(P))/3`, `temperature/10`,
`PET/3`, `log1p(area)/5` and `elevation/500`. Here `P` is precipitation. The
rolling means use the current and preceding rows in the date-sorted provider
series, with windows of seven/thirty rows and `min_periods` equal to the full
window length. They are computed before the fitting/scoring periods are selected;
rows with nonfinite features are then excluded. This gives the 20 native
predictors after removing the constant column.

`configs/acquisition-v1.json` retains provider URLs, original container hashes,
and the exact scientific member hashes used by preparation. `acquire` downloads
only absent archives and verifies them. Network endpoints are mutable; a catalogue link alone does not verify
a complete download. Provider observations and prepared input copies are not distributed here; obtain
them from the providers using the data README.

CAMELS download ZIP wrappers may be regenerated with different readme/timestamp
bytes. Strict mode requires the original container hash. Explicit
`--allow-container-repack` accepts a different wrapper only if every used
scientific member matches its strong hash; this is NOT recovery of the exact
original ZIP. A changed used member always fails. Preserve all actual acquired
bytes separately. `prepare` reads the three explicit raw archives without
extracting them or importing code from another workspace. Reconstructed arrays
should recover the declared selections and scientific features; metadata
formatting and container fingerprints need not match historical local files.

Scientific replay can compare numerical results while allowing regenerated input
and query fingerprints to differ. Exact fingerprints remain available for auditing
historical versions; neither mode establishes new-population validation.

EPA AQS/AirData ambient monitoring data are public domain. Source: U.S.
Environmental Protection Agency, daily nitrogen-dioxide summaries for 2021/2022.
Attribution does not imply endorsement. Provider permission and metadata:
https://www.epa.gov/outdoor-air-quality-data/do-i-need-request-permission-use-monitoring-data-and-graphics-airdata
https://aqs.epa.gov/aqsweb/airdata/FileFormats.html

CAMELS-GB data use Open Government Licence v3, not the description article's
separate licence. Contains data supplied by UK Centre for Ecology & Hydrology.
Copyright UK Centre for Ecology & Hydrology and University of Bristol. Contains
Environment Agency information, copyright Environment Agency and/or database
right. Derived from UK bedrock hydrogeological mapping and UK superficial
productivity mapping, BGS copyright UKRI. These notices are preserved even though
not all original supplied products are used. No provider endorsement is implied.

Coxon, G.; Addor, N.; Bloomfield, J.P.; Freer, J.; Fry, M.; Hannaford, J.;
Howden, N.J.K.; Lane, R.; Lewis, M.; Robinson, E.L.; Wagener, T.; Woods, R.
(2020). Catchment attributes and hydro-meteorological timeseries for 671 catchments
across Great Britain (CAMELS-GB). NERC Environmental Information Data Centre.
https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9
https://eidc.ac.uk/licences/ogl/plain
Dataset description: https://doi.org/10.5194/essd-12-2459-2020

Provider inputs and locally derived arrays remain outside the repository. Local
retention and content checks do not independently certify upstream measurement
ancestry or a second off-device recovery copy.

The selected figure PDFs embed DejaVu fonts. Their copyright and licence notices
are retained in `docs/embedded-font-license.txt`; the project MIT licence does
not replace those terms. The selected figures, readable tables and numerical
summaries are authored scientific outputs derived from the credited inputs,
not copies of provider observation datasets.
