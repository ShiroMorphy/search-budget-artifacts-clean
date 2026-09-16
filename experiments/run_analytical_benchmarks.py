"""
Analytical Benchmark: Known SVD Ground Truth vs. Manifold Optimizers
===================================================================
Produces Table 1 for the manuscript:
Comparing Fixed-step RGD (budgets 4/35, 12/100, 25/150) against monotone
backtracking, genuine Armijo, Riemannian Barzilai-Borwein (RBB), and Wen-Yin
Cayley.
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
    optimize_armijo,
    optimize_rbb,
    optimize_wen_yin,
    compute_cefi_with_method
)

import cefi_independent as ci

def main():
    rng = np.random.default_rng(11)
    p = 30
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    # Normal symmetric system: SVD bound is exactly achievable
    A = Q @ np.diag(np.linspace(0.9, 0.05, p)) @ Q.T
    S = np.eye(p)
    Sx = np.eye(p)

    # Analytical optimum
    ce_opt, q_opt, spec_opt = ci.cefi_svd_bound(A, S, Sx)
    print(f"ANALYTICAL OPTIMUM (SVD Ground Truth): CEFI = {ce_opt:.6f}, q* = {q_opt}\n")

    benchmarks = [
        ("Fixed 4/35", lambda A, S, s2, q, **kw: optimize_fixed(A, S, s2, q, n_restarts=4, max_iter=35, **kw)),
        ("Fixed 12/100", lambda A, S, s2, q, **kw: optimize_fixed(A, S, s2, q, n_restarts=12, max_iter=100, **kw)),
        ("Fixed 25/150", lambda A, S, s2, q, **kw: optimize_fixed(A, S, s2, q, n_restarts=25, max_iter=150, **kw)),
        ("Monotone backtracking 12/100", lambda A, S, s2, q, **kw: optimize_monotone_backtracking(A, S, s2, q, n_restarts=12, max_iter=100, **kw)),
        ("Armijo 12/100 (c1=1e-4)", lambda A, S, s2, q, **kw: optimize_armijo(A, S, s2, q, n_restarts=12, max_iter=100, **kw)),
        ("Riemannian BB 12/100", lambda A, S, s2, q, **kw: optimize_rbb(A, S, s2, q, n_restarts=12, max_iter=100, **kw)),
        ("Wen-Yin Cayley 12/100", lambda A, S, s2, q, **kw: optimize_wen_yin(A, S, s2, q, n_restarts=12, max_iter=100, **kw)),
    ]

    results = []
    print(f"{'Method / Configuration':<26} {'CEFI':>10} {'q*':>4} {'% of Opt':>10} {'Deficit':>10} {'Time (s)':>10} {'Total Steps':>12}")
    print("-" * 86)

    for name, fn in benchmarks:
        ce, q, spec, eim, details = compute_cefi_with_method(A, S, Sx, fn)
        pct = (ce / ce_opt) * 100.0
        deficit = ((ce_opt - ce) / ce_opt) * 100.0
        total_time = sum(d["elapsed_sec"] for d in details.values())
        total_steps = sum(d["total_steps"] for d in details.values())
        objective_evals = sum(d["objective_evals"] for d in details.values())
        gradient_evals = sum(d["gradient_evals"] for d in details.values())
        retraction_evals = sum(d["retraction_evals"] for d in details.values())
        
        print(f"{name:<26} {ce:10.6f} {q:4d} {pct:9.2f}% {deficit:9.2f}% {total_time:10.3f} {total_steps:12d}")
        results.append({
            "method": name,
            "cefi": ce,
            "q_star": q,
            "pct_optimal": pct,
            "deficit_pct": deficit,
            "time_sec": total_time,
            "total_steps": total_steps,
            "objective_evals": objective_evals,
            "gradient_evals": gradient_evals,
            "retraction_evals": retraction_evals
        })

    df = pd.DataFrame(results)
    df.to_csv("experiments/analytical_benchmarks.csv", index=False)
    print("\nSaved experiments/analytical_benchmarks.csv")

if __name__ == "__main__":
    main()
