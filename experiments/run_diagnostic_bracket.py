"""
Feasible-Witness Suboptimality Certificate on the Empirical Trajectory
======================================================================
Regenerates `evidence/q1_bounds_summary.csv`, the table behind Figure 2 and
behind the reported optimization deficit.

For each ridge regularization level, the estimator is run over 18
non-overlapping 500-day windows of the 30 industry portfolio returns and two
quantities are averaged across windows:

  mean_cefi         the production estimate, fixed step 12/100, maximized
                    over macro dimensions q = 1 .. p-1;
  q1_feasible_cefi  the best feasible projection witness at q = 1 obtained
                    from monotone backtracking, RBB and Wen-Yin searches.

Because the witness is itself an admissible point of the Stiefel manifold, any
excess of the witness over the production estimate certifies that the latter is
sub-optimal, without any claim about the global maximum (Proposition 3).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('experiments'))
sys.path.insert(0, os.path.abspath('evidence'))

from optimizers import (
    clean_covariance,
    compute_cefi_with_method,
    optimize_monotone_backtracking,
    optimize_fixed,
    optimize_rbb,
    optimize_wen_yin,
)
import cefi_independent as ci

LAMBDAS = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
N_WINDOWS = 18
WINDOW = 500
KAPPA = 1.0


WITNESS_METHODS = (optimize_monotone_backtracking, optimize_rbb, optimize_wen_yin)


def feasible_q1_witness(A, S_eps, Sigma_x, kappa=KAPPA):
    """
    Causal emergence density at q = 1 for the best witness found by the three
    adaptive Riemannian methods benchmarked in the paper, all at the production
    restart budget.  Any feasible point certifies sub-optimality, so taking the
    best of the three only sharpens the certificate; doubling the budget to
    24 restarts and 200 iterations changes the result in the seventh digit.
    """
    p = A.shape[0]
    scale = max(np.trace(Sigma_x) / p, 1e-12)
    s2 = (kappa ** 2) * scale
    Sc = clean_covariance(S_eps, scale)
    eim = ci.ei_micro(A, Sc, s2)
    best = max(m(A, Sc, s2, 1, n_restarts=12, max_iter=100,
                 Sigma_x=Sigma_x)["best_f"] for m in WITNESS_METHODS)
    return best / 1.0 - eim / p


def main() -> int:
    df = pd.read_csv("data/ff30_daily_returns.csv", parse_dates=["Date"],
                     index_col="Date")
    X = df.values
    wins = [X[i * WINDOW:(i + 1) * WINDOW] for i in range(N_WINDOWS)]
    if len(wins[-1]) < WINDOW:
        raise RuntimeError("not enough observations for 18 non-overlapping windows")

    print(f"{'lambda0':>10} {'q1_feasible':>14} {'mean_cefi':>14} "
          f"{'deficit':>10} {'sec':>7}")
    print("-" * 60)

    records = []
    for lam in LAMBDAS:
        t0 = time.perf_counter()
        prod, feas = [], []
        for w in wins:
            A, S_eps = ci.fit_var1(w, lam)
            Sx = np.cov(w, rowvar=False)
            ce, _, _, _, _ = compute_cefi_with_method(
                A, S_eps, Sx,
                lambda A_, S_, s2_, q_, **kw: optimize_fixed(
                    A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw))
            prod.append(ce)
            feas.append(feasible_q1_witness(A, S_eps, Sx))

        mean_cefi = float(np.mean(prod))
        q1_feas = float(np.mean(feas))
        ratio = q1_feas / mean_cefi if mean_cefi != 0 else np.inf
        records.append({"lambda_0": lam, "q1_feasible_cefi": q1_feas,
                        "mean_cefi": mean_cefi, "bound_to_reported": ratio})
        print(f"{lam:>10g} {q1_feas:>14.8f} {mean_cefi:>14.8f} "
              f"{ratio:>9.2f}x {time.perf_counter() - t0:>6.1f}")

    out = pd.DataFrame(records)
    os.makedirs("evidence", exist_ok=True)
    out.to_csv("evidence/q1_bounds_summary.csv", index=False)
    peak = out.loc[out["bound_to_reported"].idxmax()]
    print(f"\npeak deficit {peak['bound_to_reported']:.1f}x "
          f"at lambda_0 = {peak['lambda_0']:g}")
    print("Saved evidence/q1_bounds_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
