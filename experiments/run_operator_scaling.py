"""
Operator Scaling Law, Breakdown Point, and Normalization Control
================================================================
Investigates the scaling behavior of the objective and Riemannian gradient
as a function of operator norm ||A||, finds the breakdown point of 12/100,
and performs the normalization control experiment.
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
    compute_cefi_with_method,
    euclidean_gradient,
    project_tangent,
    objective
)
import cefi_independent as ci

def main():
    rng = np.random.default_rng(42)
    p = 30
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    # Base symmetric matrix with unit spectral radius
    A_base = Q @ np.diag(np.linspace(1.0, 0.1, p)) @ Q.T
    S = np.eye(p)
    Sx = np.eye(p)

    # Scale factors alpha from 0.005 to 1.5 (18 logarithmically spaced points)
    alphas = np.logspace(np.log10(0.005), np.log10(1.5), 18)

    results = []
    print(f"{'alpha':>8} {'||A||_F':>8} {'SVD_opt':>10} {'Fixed_12_100':>12} {'Attn_Fixed':>11} {'q*_fix':>6} {'Adapt_12_100':>12} {'Attn_Adp':>9} {'Norm_q1_Attn':>13} {'GradNorm_W0':>12}")
    print("-" * 109)

    # Fixed test point W0 on St(1, p) to track initial gradient norm
    W0, _ = np.linalg.qr(rng.standard_normal((p, 1)))
    W0 = W0.T

    for alpha in alphas:
        A = alpha * A_base
        frob = np.linalg.norm(A, 'fro')
        
        # Analytical optimum
        ce_opt, q_opt, spec_opt = ci.cefi_svd_bound(A, S, Sx)

        # Track initial gradient norm for q=1
        scale = max(np.trace(Sx) / p, 1e-12)
        s2 = scale
        ge0 = euclidean_gradient(W0, A, S, s2)
        gr0 = project_tangent(W0, ge0)
        gnorm0 = np.linalg.norm(gr0)

        # Fixed 12/100
        ce_fix, q_fix, _, _, _ = compute_cefi_with_method(
            A, S, Sx,
            lambda A_, S_, s2_, q_, **kw: optimize_fixed(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        attn_fix = ce_fix / ce_opt if ce_opt > 0 else 1.0

        # Monotone backtracking 12/100
        ce_adp, q_adp, _, _, _ = compute_cefi_with_method(
            A, S, Sx,
            lambda A_, S_, s2_, q_, **kw: optimize_monotone_backtracking(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        attn_adp = ce_adp / ce_opt if ce_opt > 0 else 1.0

        # Normalization control experiment:
        # Scale A to unit Frobenius norm, optimize with fixed 12/100, then evaluate at true A
        norm_factor = frob
        A_norm = A / norm_factor
        res_norm = optimize_fixed(A_norm, S, s2, 1, n_restarts=12, max_iter=100)
        W_norm = res_norm["best_W"]
        f_norm = objective(W_norm, A, S, s2)
        eim = ci.ei_micro(A, S, s2) / p
        ce_norm_q1 = f_norm / 1.0 - eim
        ce_opt_q1 = spec_opt[1]
        attn_norm_q1 = ce_norm_q1 / ce_opt_q1 if ce_opt_q1 > 0 else 1.0

        print(f"{alpha:8.4f} {frob:8.4f} {ce_opt:10.6f} {ce_fix:12.6f} {attn_fix:10.2%} {q_fix:6d} {ce_adp:12.6f} {attn_adp:8.2%} {attn_norm_q1:12.2%} {gnorm0:12.4e}")

        results.append({
            "alpha": alpha,
            "frob_norm": frob,
            "ce_opt": ce_opt,
            "q_opt": q_opt,
            "ce_fixed": ce_fix,
            "q_fixed": q_fix,
            "attn_fixed": attn_fix,
            "ce_adaptive": ce_adp,
            "q_adaptive": q_adp,
            "attn_adaptive": attn_adp,
            "ce_norm_q1": ce_norm_q1,
            "attn_norm_q1": attn_norm_q1,
            "grad_norm_W0": gnorm0
        })

    df = pd.DataFrame(results)
    df.to_csv("experiments/operator_scaling.csv", index=False)
    print("\nSaved experiments/operator_scaling.csv")

if __name__ == "__main__":
    main()
