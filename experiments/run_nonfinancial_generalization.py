"""
Generalization Beyond Finance:
1. Synthetic Coupled Network (Coupling Strength sweep g)
2. Finite Sample Size / Window Length sweep T
3. Real Geophysical Climate System (NOAA Monthly Oscillation Indices)
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

def run_coupling_sweep():
    print("\n--- 1. SYNTHETIC COUPLED NETWORK (Coupling Strength g) ---")
    p = 20
    rng = np.random.default_rng(101)
    
    # Random directed network adjacency C with spectral radius normalized to 1.0
    C = rng.standard_normal((p, p))
    np.fill_diagonal(C, 0.0)
    rad = max(abs(np.linalg.eigvals(C)))
    C = C / rad
    
    # Baseline self-decay
    a = 0.15
    S_eps = np.eye(p)
    Sx = np.eye(p)
    
    g_values = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
    records = []
    
    print(f"{'g':>6} {'||A||_F':>8} {'SVD_Bound':>12} {'Fixed_12_100':>14} {'Adapt_12_100':>14} {'Attn_Fixed':>11} {'Attn_Adapt':>11}")
    print("-" * 80)
    
    for g in g_values:
        A = a * np.eye(p) + g * C
        frob = np.linalg.norm(A, 'fro')
        
        ce_svd, q_svd, _ = ci.cefi_svd_bound(A, S_eps, Sx)
        ce_fix, q_fix, _, _, _ = compute_cefi_with_method(
            A, S_eps, Sx,
            lambda A_, S_, s2_, q_, **kw: optimize_fixed(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        ce_adp, q_adp, _, _, _ = compute_cefi_with_method(
            A, S_eps, Sx,
            lambda A_, S_, s2_, q_, **kw: optimize_monotone_backtracking(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        
        attn_fix = ce_fix / ce_svd if ce_svd > 0 else 1.0
        attn_adp = ce_adp / ce_svd if ce_svd > 0 else 1.0
        
        print(f"{g:6.2f} {frob:8.4f} {ce_svd:12.6f} {ce_fix:14.6f} {ce_adp:14.6f} {attn_fix:10.2%} {attn_adp:10.2%}")
        records.append({
            "g": g,
            "frob_norm": frob,
            "ce_svd": ce_svd,
            "ce_fixed": ce_fix,
            "ce_adaptive": ce_adp,
            "attn_fixed": attn_fix,
            "attn_adaptive": attn_adp,
            "q_svd": q_svd,
            "q_fixed": q_fix,
            "q_adaptive": q_adp
        })
        
    df = pd.DataFrame(records)
    df.to_csv("experiments/coupled_network_sweep.csv", index=False)
    return df

def run_sample_size_sweep():
    print("\n--- 2. FINITE SAMPLE SIZE / WINDOW LENGTH SWEEP (T) ---")
    p = 15
    rng = np.random.default_rng(202)
    
    # Ground truth VAR(1) process
    A_true = 0.4 * np.eye(p)
    for i in range(p - 1):
        A_true[i, i + 1] = 0.25
        A_true[i + 1, i] = -0.15
    
    T_max = 5000
    burn_in = 500
    X_sim = np.zeros((T_max + burn_in, p))
    for t in range(1, T_max + burn_in):
        X_sim[t] = A_true @ X_sim[t - 1] + rng.standard_normal(p)
    X_sim = X_sim[burn_in:]
    
    T_values = [60, 120, 250, 500, 1000, 2500, 5000]
    records = []
    
    print(f"{'T':>6} {'||A_hat||_F':>12} {'SVD_Bound':>12} {'Fixed_12_100':>14} {'Adapt_12_100':>14} {'Attn_Fixed':>11} {'Attn_Adapt':>11}")
    print("-" * 84)
    
    for T in T_values:
        sub_X = X_sim[:T]
        A_hat, S_hat = ci.fit_var1(sub_X, lam0=0.1) # standard shrinkage
        Sx_hat = np.cov(sub_X, rowvar=False)
        frob = np.linalg.norm(A_hat, 'fro')
        
        ce_svd, q_svd, _ = ci.cefi_svd_bound(A_hat, S_hat, Sx_hat)
        ce_fix, q_fix, _, _, _ = compute_cefi_with_method(
            A_hat, S_hat, Sx_hat,
            lambda A_, S_, s2_, q_, **kw: optimize_fixed(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        ce_adp, q_adp, _, _, _ = compute_cefi_with_method(
            A_hat, S_hat, Sx_hat,
            lambda A_, S_, s2_, q_, **kw: optimize_monotone_backtracking(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        
        attn_fix = ce_fix / ce_svd if ce_svd > 0 else 1.0
        attn_adp = ce_adp / ce_svd if ce_svd > 0 else 1.0
        
        print(f"{T:6d} {frob:12.4f} {ce_svd:12.6f} {ce_fix:14.6f} {ce_adp:14.6f} {attn_fix:10.2%} {attn_adp:10.2%}")
        records.append({
            "T": T,
            "frob_norm": frob,
            "ce_svd": ce_svd,
            "ce_fixed": ce_fix,
            "ce_adaptive": ce_adp,
            "attn_fixed": attn_fix,
            "attn_adaptive": attn_adp,
            "q_svd": q_svd,
            "q_fixed": q_fix,
            "q_adaptive": q_adp
        })
        
    df = pd.DataFrame(records)
    df.to_csv("experiments/sample_size_sweep.csv", index=False)
    return df

def run_climate_empirical():
    print("\n--- 3. REAL GEOPHYSICAL CLIMATE SYSTEM (NOAA Monthly Indices) ---")
    df_clim = pd.read_csv("data/noaa_climate_indices.csv", index_col=0)
    X = df_clim.values
    p = X.shape[1]
    
    lambdas = [1e-4, 0.01, 0.1, 0.5, 2.0, 5.0]
    records = []
    
    print(f"{'lambda0':>10} {'||A||_F':>10} {'Fixed_12_100':>14} {'Adapt_12_100':>14} {'SVD_Bound':>12} {'Ratio_Adp/Fix':>14}")
    print("-" * 78)
    
    for lam in lambdas:
        A, S_eps = ci.fit_var1(X, lam0=lam)
        Sx = np.cov(X, rowvar=False)
        frob = np.linalg.norm(A, 'fro')
        
        ce_svd, q_svd, _ = ci.cefi_svd_bound(A, S_eps, Sx)
        ce_fix, q_fix, _, _, _ = compute_cefi_with_method(
            A, S_eps, Sx,
            lambda A_, S_, s2_, q_, **kw: optimize_fixed(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        ce_adp, q_adp, _, _, _ = compute_cefi_with_method(
            A, S_eps, Sx,
            lambda A_, S_, s2_, q_, **kw: optimize_monotone_backtracking(A_, S_, s2_, q_, n_restarts=12, max_iter=100, **kw)
        )
        
        ratio = ce_adp / ce_fix if ce_fix > 0 else np.nan
        print(f"{lam:10.4f} {frob:10.4f} {ce_fix:14.6f} {ce_adp:14.6f} {ce_svd:12.6f} {ratio:13.1f}x")
        
        records.append({
            "lambda_0": lam,
            "frob_norm": frob,
            "ce_fixed": ce_fix,
            "ce_adaptive": ce_adp,
            "ce_svd": ce_svd,
            "ratio_adapt_fixed": ratio,
            "q_fixed": q_fix,
            "q_adaptive": q_adp,
            "q_svd": q_svd
        })
        
    df = pd.DataFrame(records)
    df.to_csv("experiments/climate_generalization.csv", index=False)
    return df

def main():
    run_coupling_sweep()
    run_sample_size_sweep()
    run_climate_empirical()

if __name__ == "__main__":
    main()
