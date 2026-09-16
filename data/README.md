# Data

## `ff30_daily_returns.csv` (not included)

Daily value-weighted returns of the 30 industry portfolios, 1990-01-02 to
2026-06-30, 9190 trading days, expressed as decimals.

**This file is deliberately not redistributed here.** The Kenneth R. French Data
Library states that its contents are the property of Ken French and Dimensional
Fund Advisors and that use in part or whole requires their permission. The file
is therefore obtained from the source:

```
python3 data/download_ff30.py
```

The script downloads `30_Industry_Portfolios_daily_CSV.zip` from the French Data
Library, extracts the `Average Value Weighted Returns -- Daily` block, restricts
it to the paper's window, converts percent to decimals, and writes the CSV. It
then prints the SHA-256 of what it wrote and compares it against

```
daa0e590dc7c41c704044341ef9c25748e5401f02587019a6d848a60ac4d1daf
```

which is the hash of the vintage retrieved on 14 September 2026, behind every
number reported in the paper. If the hashes match, the reproduction is exact.

If they do not, that is expected over time rather than an error: the French
library is restated periodically as the underlying CRSP data are revised. A
retrieval on 14 September 2026 differed from an earlier one in 32 cells out of
275,700, with a maximum absolute difference of 0.0034, and every published
result was unaffected to the precision reported. The script therefore prints a
notice and exits zero; re-run `reproduce_paper1.sh` to recompute the empirical
results on your own retrieval and compare them against the published values.

The retrieval script has been validated end to end by round trip: the published
CSV was re-encoded into the raw layout of the French daily file (percent returns,
`YYYYMMDD` dates, header block, trailing equal-weighted block), passed through
the script, and the output was byte-for-byte identical to the original, with a
matching SHA-256.

Source: Kenneth R. French Data Library,
<https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>.
The underlying returns are constructed from CRSP data.

The empirical sweep uses the first 9,000 observations as 18 consecutive,
non-overlapping windows of 500 trading days:

```
1990-01-02--1991-12-20, 1991-12-23--1993-12-13,
1993-12-14--1995-12-05, 1995-12-06--1997-11-25,
1997-11-26--1999-11-19, 1999-11-22--2001-11-19,
2001-11-20--2003-11-13, 2003-11-14--2005-11-08,
2005-11-09--2007-11-05, 2007-11-06--2009-10-29,
2009-10-30--2011-10-24, 2011-10-25--2013-10-21,
2013-10-22--2015-10-15, 2015-10-16--2017-10-10,
2017-10-11--2019-10-07, 2019-10-08--2021-09-30,
2021-10-01--2023-09-27, 2023-09-28--2025-09-25
```

The remaining 190 observations, from 2025-09-26 through 2026-06-30, are not
used in the windowed empirical sweep.

## `noaa_climate_indices.csv` (included)

Monthly values of ten climate oscillation indices (SOI, NAO, PDO, AO, AMO,
NINO34, PNA, WP, EA, EPO), 1951-01 onward, from the U.S. National Oceanic and
Atmospheric Administration. Works of the U.S. federal government are not subject
to domestic copyright protection, so this file is included directly.
