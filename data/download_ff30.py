#!/usr/bin/env python3
"""
Rebuild `data/ff30_daily_returns.csv` from the Kenneth R. French Data Library.

The raw data are NOT redistributed with this repository.  The French Data
Library states that its contents are the property of Ken French and Dimensional
Fund Advisors and that use in part or whole requires permission, so this script
downloads the file directly from the source instead.

Output: data/ff30_daily_returns.csv
    Column `Date` plus the 30 industry portfolios, daily value-weighted
    returns expressed as decimals (not percent), from 1990-01-02 onward.

Usage:
    python3 data/download_ff30.py [--end 2026-06-30]

The script prints the SHA-256 of the file it writes and compares it against the
hash of the file used in the paper.  A mismatch is expected if French has since
revised the series or if you choose a different end date; it is NOT expected
otherwise, and it means the downstream numbers will not match the paper.
"""

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile

import pandas as pd

URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
       "30_Industry_Portfolios_daily_CSV.zip")

# SHA-256 of the vintage used for every number reported in the paper:
# retrieved 14 September 2026, 9190 rows, 1990-01-02 to 2026-06-30.
# The French Data Library is periodically restated as the underlying CRSP data
# are revised, so a later retrieval may differ in a small number of cells.
PAPER_SHA256 = "daa0e590dc7c41c704044341ef9c25748e5401f02587019a6d848a60ac4d1daf"

START = "1990-01-02"


def fetch() -> str:
    """Download the zip and return the decoded text of the CSV inside it."""
    with urllib.request.urlopen(URL, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name).decode("latin-1")


def parse_first_block(text: str) -> pd.DataFrame:
    """
    The French daily file contains several stacked blocks.  The first one is
    'Average Value Weighted Returns -- Daily', which is what the paper uses.

    A data row starts with an 8-digit date (YYYYMMDD).  The header row is the
    last non-data line before the first data row.  The block ends at the first
    blank or non-data line after it.
    """
    lines = text.splitlines()
    header = None
    rows = []
    started = False
    for line in lines:
        first = line.split(",")[0].strip()
        is_data = len(first) == 8 and first.isdigit()
        if not started:
            if is_data:
                started = True
                rows.append(line)
            elif line.strip():
                header = line
        else:
            if is_data:
                rows.append(line)
            else:
                break
    if header is None or not rows:
        raise RuntimeError("could not locate the daily returns block")

    cols = [c.strip() for c in header.split(",")]
    cols[0] = "Date"
    df = pd.read_csv(io.StringIO("\n".join([",".join(cols)] + rows)))
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-06-30",
                    help="last date to keep (default: the paper's end date)")
    ap.add_argument("--out", default="data/ff30_daily_returns.csv")
    args = ap.parse_args()

    df = parse_first_block(fetch())
    df = df[(df["Date"] >= START) & (df["Date"] <= args.end)].copy()

    value_cols = [c for c in df.columns if c != "Date"]
    # French publishes percent returns and codes missing values as -99.99.
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    df[value_cols] = df[value_cols].mask(df[value_cols] <= -99.0)
    if df[value_cols].isna().any().any():
        n = int(df[value_cols].isna().sum().sum())
        print(f"warning: {n} missing observations coded by French as -99.99",
              file=sys.stderr)
    df[value_cols] = df[value_cols] / 100.0

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(args.out, index=False)

    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"wrote {args.out}: {len(df)} rows, {len(value_cols)} portfolios, "
          f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"sha256   {digest}")
    print(f"expected {PAPER_SHA256}")
    if digest == PAPER_SHA256:
        print("MATCH: this is byte-for-byte the file used in the paper.")
        return 0
    print("NOTICE: this retrieval differs from the vintage used in the paper.")
    print("The French Data Library is restated periodically, so this is "
          "expected over time.  Re-run reproduce_paper1.sh to recompute the "
          "empirical results on the retrieval you just made, and compare them "
          "against the published values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
