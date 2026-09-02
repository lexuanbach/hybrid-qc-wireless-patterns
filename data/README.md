# Bundled Abilene trace summary

`abilene_x01_summary.csv` contains 240 evenly spaced windows derived from the
real origin--destination traffic matrices in Abilene/Internet2 data file
`X01.gz` (week beginning 2004-03-01). Each row reports aggregate statistics
from the 144 real OD entries; estimated gravity/tomography columns are not
used. Loads are divided by the median total load of all 2,016 source windows.

Source-file SHA-256:
`d996a9115d2815eec158d5055452deb389f5546a96e0dfcbfae5c22a40372558`.

The summary is bundled so the artifact runs without another workspace. The
paper and artifact label only the traffic quantities as measured. QPU queue,
calibration, price, and injected fault quantities remain modeled.

`abilene_three_weeks_summary.csv` applies the same aggregation to X01, X02,
and X03 (week starts March 1, March 8, and April 2, 2004), retaining 240 evenly
spaced windows per measured week.  `abilene_three_weeks_provenance.json`
records each official UT Austin URL and SHA-256.  The raw weekly archives are
not bundled; `examples/prepare_abilene_weeks.py` recreates the summary.

## Bundled NetData 5G summary

`netdata_5g_summary.csv` contains 48 half-hour aggregates over 12 cells from
the public NetData weekday performance file.  The deterministic preprocessing
selects the four base stations with the largest mean traffic among stations
with at least three cells, then the three largest-traffic cells at each
station.  Each row contains mean/p95 physical-resource-block (PRB) use and
total traffic and users.  `netdata_5g_provenance.json` records the upstream
URL, commit, raw SHA-256, selected identifiers, and aggregation.  Because the
upstream repository declared no license at access time, the 76 MB raw CSV is
not redistributed; `examples/prepare_netdata_summary.py` recreates the compact
summary when supplied with that file.
`netdata_5g_cells.csv` contains the corresponding 576 cell--window rows used
for the P2 high-load prediction task; it retains only mean PRB use, traffic,
users, identifiers, and half-hour timestamp.

The NTN study combines these measured RAN-load fields with explicitly modeled
LEO geometry, path loss, shadowing, Rician fading, capacity, QPU queue, and
quantum quality.  No radio, satellite, QPU, or end-to-end field outcome is
labeled measured.
