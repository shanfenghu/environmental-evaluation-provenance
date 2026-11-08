# Inputs and attribution

The prepared-input bundle contains the exact two numeric archives used by the
native recipe: `air_quality_2021_2022.npz` and `hydrology.npz`. Their source-indexed
arrays contain earlier predictors/responses (`x`, `y`), later predictors/responses
(`z`, `w`) and date indices (`t`, `s`). Source order, identities, coordinates,
cell assignments and row counts are retained in the matching metadata. The
first prepared feature is constant and is dropped by the native loader; the
resulting native predictor dimensions are 13 and 20. Only hydrological fitting
responses transform to log1p. Do not replace these pinned inputs silently.

EPA nitrogen-dioxide summaries retain 353 sites, fitting in 2021 and scoring in
2022. Geographic bounds are 25–50 degrees latitude and -125 to -66 longitude;
days require at least 18 observations. A single POC is chosen by earlier-year
completeness, with deterministic tie-breaking. Ambiguous site dates and coordinate
drift are excluded; each year needs at least 300 retained valid days. Negative
reported means are retained. Geographic cuts are latitude 39 and longitude -100.

The original CAMELS-GB 2020 release supplies 618 retained catchments. Fitting is
1995-10-01–2005-09-30 and scoring 2005-10-01–2015-09-30, with at least 90% valid
days in each period. Negative flow and nonfinite response/predictor rows are
invalid. Geographic cuts are latitude 53.5 and longitude -2.5. Fixed coordinate
and seasonal terms are supplemented by transformed precipitation, seven-/thirty-
day precipitation means, temperature, potential evapotranspiration, area and
elevation. Contemporary observed forcings make this retrospective assessment;
they are not an operational forecasting promise. Completeness filters do not
establish representative environmental sampling.

`configs/acquisition-v1.json` retains provider URLs, original container hashes,
and the exact scientific member hashes used by preparation. `acquire` downloads
only absent archives and verifies them. Network endpoints are mutable and were
not freshly downloaded for this candidate. The prepared bundle avoids relying
on a current server to recover the original numerical inputs.

CAMELS download ZIP wrappers may be regenerated with different readme/timestamp
bytes. Strict mode requires the original container hash. Explicit
`--allow-container-repack` accepts a different wrapper only if every used
scientific member matches its strong hash; this is NOT recovery of the exact
original ZIP. A changed used member always fails. Preserve all actual acquired
bytes separately. `prepare` reads the three explicit raw archives without
extracting them or importing code from another workspace. Reconstructed arrays
should match names, dtypes, shapes and values of the prepared bundle; metadata
formatting and byte fingerprints need not match after neutralisation.

The bundled archives supply the frozen byte-version for exact query-signature
replay. A content-identical repack with different NPZ bytes has a different
input version; it does not justify claiming unchanged query identifiers.

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

The bundled data are derived, filtered arrays from that original release. Exact
local retention and content verification do not independently certify upstream
measurement ancestry or a second off-device recovery copy.
