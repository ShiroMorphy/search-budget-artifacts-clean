"""
Grid Sweep: Learning Rate vs. Number of Iterations
=================================================
Disentangles the effect of step size (lr) from iteration budget (K)
at a contracted operator scale (alpha = 0.10, ||A||_F ~ 0.34).
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('experiments'))
sys.path.insert(0, os.path.abspath('evidence'))

from optimizers import (
    optimize_fixed,
    optimize_monotone_backtracking,
    compute_cefi_with_method
)
import cefi_independent as ci

def main():
    rng = np.random.default_rng(42)
    p = 30
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    alpha = 0.10
    A = alpha * (Q @ np.diag(np.linspace(1.0, 0.1, p)) @ Q.T)
    S = np.eye(p)
    Sx = np.eye(p)

    ce_opt, q_opt, spec_opt = ci.cefi_svd_bound(A, S, Sx)
    print(f"Contracted Operator alpha={alpha:.2f}, Analytical Optimum CEFI = {ce_opt:.6f}, q* = {q_opt}\n")

    lrs = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
    iters = [25, 50, 100, 200, 500, 1000]

    grid_results = []
    print(f"{'LR (eta)':>10} {'K (iter)':>10} {'CEFI':>12} {'% of Opt':>10} {'q*':>4}")
    print("-" * 52)

    for lr in lrs:
        for k in iters:
            # We test q=1 for speed and clarity
            scale = max(np.trace(Sx) / p, 1e-12)
            s2 = scale
            res = optimize_fixed(A, S, s2, 1, n_restarts=12, max_iter=k, lr=lr)
            eim = ci.ei_micro(A, S, s2) / p
            ce_val = res["best_f"] / 1.0 - eim
            pct = (ce_val / ce_opt) * 100.0
            print(f"{lr:10.3f} {k:10d} {ce_val:12.6f} {pct:9.2f}% {1:4d}")
            grid_results.append({
                "lr": lr,
                "iterations": k,
                "cefi_q1": ce_val,
                "pct_opt": pct
            })

    df = pd.DataFrame(grid_results)
    df.to_csv("experiments/lr_vs_iter_sweep.csv", index=False)
    print("\nSaved experiments/lr_vs_iter_sweep.csv")

if __name__ == "__main__":
    main()
