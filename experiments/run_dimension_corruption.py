"""
Macro-Dimension Selection Corruption Artifact
=============================================
Demonstrates how search-budget collapse corrupts the selected macro-dimension q*,
falsely shifting it from true low-dimensional macro structure (q*=1)
to high-dimensional boundary scales (q* ~ 27-29).
Evaluates both on the synthetic normal system and across empirical windows.
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
    # 1. Synthetic controlled experiment:
    # A has rank-1 dominant macro structure with small decaying background modes
    rng = np.random.default_rng(11)
    p = 30
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    diag_modes = np.linspace(0.9, 0.05, p)
    A_sym = Q @ np.diag(diag_modes) @ Q.T
    S = np.eye(p)
    Sx = np.eye(p)

    # Vary contraction alpha
    alphas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    synthetic_records = []

    print("=== SYNTHETIC SYSTEM DIMENSION CORRUPTION ===")
    print(f"{'alpha':>8} {'q*_true':>8} {'q*_fixed_4_35':>14} {'q*_fixed_12_100':>16} {'q*_adapt_12_100':>16}")
    print("-" * 66)

    for a in alphas:
        A = a * A_sym
        _, q_opt, _ = ci.cefi_svd_bound(A, S, Sx)
        _, q_f4, _, _, _ = compute_cefi_with_method(A, S, Sx, lambda A_, S_, s2_, q_, **kw: optimize_fixed(A_, S_, s2_, q_, n_restarts=4, max_iter=35, **kw))
        _, q_f12, _, _, _ = compute_cefi_with_method(A, S, Sx, lambda A_, S_, s2_, q_, **kw: optimize_fixed(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw))
        _, q_adp, _, _, _ = compute_cefi_with_method(A, S, Sx, lambda A_, S_, s2_, q_, **kw: optimize_monotone_backtracking(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw))

        print(f"{a:8.2f} {q_opt:8d} {q_f4:14d} {q_f12:16d} {q_adp:16d}")
        synthetic_records.append({
            "alpha": a,
            "q_true": q_opt,
            "q_fixed_4_35": q_f4,
            "q_fixed_12_100": q_f12,
            "q_adapt_12_100": q_adp
        })

    pd.DataFrame(synthetic_records).to_csv("experiments/synthetic_q_corruption.csv", index=False)

    # 2. Empirical windows: compare q* profiles across regularization lambda_0
    df_emp = pd.read_csv("data/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    X = df_emp.values
    # 18 non-overlapping windows of 500 days
    wins = [X[i*500:(i+1)*500] for i in range(18)]
    
    lambdas = [1e-4, 0.01, 0.1, 0.2, 0.5, 1.0, 3.0]
    emp_records = []

    print("\n=== EMPIRICAL SYSTEM DIMENSION CORRUPTION (18 non-overlapping windows) ===")
    print(f"{'lambda0':>10} {'Modal_q*_Fixed12':>18} {'Frac_q_le_4_Fixed':>18} {'Modal_q*_Adapt12':>18} {'Modal_q*_SVD':>14}")
    print("-" * 82)

    for lam in lambdas:
        q_fixed_list = []
        q_adapt_list = []
        q_svd_list = []

        for w in wins:
            A, S_eps = ci.fit_var1(w, lam)
            Sx_w = np.cov(w, rowvar=False)
            
            # SVD bound
            _, q_svd, _ = ci.cefi_svd_bound(A, S_eps, Sx_w)
            q_svd_list.append(q_svd)
            
            # Fixed 12/100
            _, q_fix, _, _, _ = compute_cefi_with_method(
                A, S_eps, Sx_w,
                lambda A_, S_, s2_, q_, **kw: optimize_fixed(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
            )
            q_fixed_list.append(q_fix)

            # Monotone backtracking 12/100
            _, q_adp, _, _, _ = compute_cefi_with_method(
                A, S_eps, Sx_w,
                lambda A_, S_, s2_, q_, **kw: optimize_monotone_backtracking(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
            )
            q_adapt_list.append(q_adp)

        # Compute modes
        from scipy.stats import mode
        mod_fix = int(mode(q_fixed_list, keepdims=False)[0])
        mod_adp = int(mode(q_adapt_list, keepdims=False)[0])
        mod_svd = int(mode(q_svd_list, keepdims=False)[0])
        frac_fix = float(np.mean([1 if q <= 4 else 0 for q in q_fixed_list]))

        print(f"{lam:10.4f} {mod_fix:18d} {frac_fix:18.1%} {mod_adp:18d} {mod_svd:14d}")
        emp_records.append({
            "lambda_0": lam,
            "modal_q_fixed": mod_fix,
            "frac_q_le_4_fixed": frac_fix,
            "modal_q_adapt": mod_adp,
            "modal_q_svd": mod_svd,
            "q_fixed_list": str(q_fixed_list),
            "q_adapt_list": str(q_adapt_list),
            "q_svd_list": str(q_svd_list)
        })

    pd.DataFrame(emp_records).to_csv("experiments/empirical_q_corruption.csv", index=False)
    print("\nSaved experiments/synthetic_q_corruption.csv and experiments/empirical_q_corruption.csv")

if __name__ == "__main__":
    main()
