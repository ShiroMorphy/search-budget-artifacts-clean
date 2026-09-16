#!/usr/bin/env python3
"""Rebuild the empirical q=1 feasible-witness diagnostic evidence.

This script is the single source of truth for Figure 2. For each
regularization level and each of 18 non-overlapping 500-day FF30 windows it
stores (i) the audited fixed-step production estimate and (ii) a feasible q=1
witness obtained with monotone backtracking. The aggregate CSV used by the
plot and the stored witness projections are written in the same run.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evidence"))
import cefi_independent as ci


LAMBDAS = (1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
N_WINDOWS = 18
WINDOW_LENGTH = 500
N_RESTARTS = 12
MAX_ITER = 100
WITNESS_RESTARTS = 24
WITNESS_MAX_ITER = 200


def evaluate_window(window, lambda_0):
    """Return the audited estimate and an admissible q=1 witness."""
    A, Sigma_eps = ci.fit_var1(window, lambda_0)
    Sigma_x = np.cov(window, rowvar=False)

    reported, q_reported, _, _ = ci.cefi(
        A,
        Sigma_eps,
        Sigma_x,
        n_restarts=N_RESTARTS,
        max_iter=MAX_ITER,
    )

    scale = max(np.trace(Sigma_x) / A.shape[0], 1e-12)
    Sigma_clean = ci._clean(Sigma_eps, scale)
    micro_density = ci.ei_micro(A, Sigma_clean, scale) / A.shape[0]
    objective, W = ci.optimize_q(
        A,
        Sigma_clean,
        scale,
        q=1,
        n_restarts=WITNESS_RESTARTS,
        max_iter=WITNESS_MAX_ITER,
        adaptive=True,
        Sigma_x=Sigma_x,
    )
    witness = objective - micro_density
    return A, Sigma_eps, Sigma_x, W, reported, q_reported, witness


def main():
    returns = pd.read_csv(ROOT / "data" / "ff30_daily_returns.csv", parse_dates=["Date"])
    values = returns.drop(columns="Date").to_numpy()
    windows = [values[i * WINDOW_LENGTH : (i + 1) * WINDOW_LENGTH] for i in range(N_WINDOWS)]
    if any(window.shape[0] != WINDOW_LENGTH for window in windows):
        raise ValueError("The FF30 input does not contain 18 complete 500-day windows.")

    records = []
    stored = {"lambdas": [], "window_ids": [], "A": [], "Sigma_eps": [], "Sigma_x": [], "W": []}
    for lambda_0 in LAMBDAS:
        for window_id, window in enumerate(windows):
            A, Sigma_eps, Sigma_x, W, reported, q_reported, witness = evaluate_window(window, lambda_0)
            records.append(
                {
                    "lambda_0": lambda_0,
                    "window_id": window_id,
                    "reported_cefi": reported,
                    "reported_q_star": q_reported,
                    "q1_feasible_cefi": witness,
                    "absolute_witness_gap": witness - reported,
                }
            )
            stored["lambdas"].append(lambda_0)
            stored["window_ids"].append(window_id)
            stored["A"].append(A)
            stored["Sigma_eps"].append(Sigma_eps)
            stored["Sigma_x"].append(Sigma_x)
            stored["W"].append(W.ravel())
        print(f"completed lambda_0={lambda_0:g}", flush=True)

    per_window = pd.DataFrame.from_records(records)
    summary = (
        per_window.groupby("lambda_0", as_index=False)
        .agg(q1_feasible_cefi=("q1_feasible_cefi", "mean"), mean_cefi=("reported_cefi", "mean"))
        .sort_values("lambda_0")
    )
    summary["bound_to_reported"] = summary["q1_feasible_cefi"] / summary["mean_cefi"]

    evidence_dir = ROOT / "evidence"
    per_window.to_csv(evidence_dir / "q1_bounds_by_window.csv", index=False)
    summary.to_csv(evidence_dir / "q1_bounds_summary.csv", index=False)
    np.savez(evidence_dir / "feasible_projection_witnesses.npz", **{key: np.asarray(value) for key, value in stored.items()})

    at_peak = per_window[np.isclose(per_window["lambda_0"], 0.5)].copy()
    gaps = at_peak["absolute_witness_gap"].to_numpy()
    print(
        "lambda_0=0.5: "
        f"ratio_of_means={summary.loc[np.isclose(summary.lambda_0, 0.5), 'bound_to_reported'].iloc[0]:.6f}; "
        f"witness_exceeds_reported={int((gaps > 0).sum())}/{len(gaps)}; "
        f"median_gap={np.median(gaps):.12f}; "
        f"iqr=({np.percentile(gaps, 25):.12f}, {np.percentile(gaps, 75):.12f})"
    )


if __name__ == "__main__":
    main()
