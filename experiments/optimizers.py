"""
Riemannian Optimization Engines for Continuous Causal Emergence
==============================================================
Provides independent, robust implementations of Stiefel manifold optimization
algorithms for effective information (EI) maximization:
1. Fixed-step Riemannian gradient ascent (audited production baseline).
2. Monotone Riemannian backtracking with best-W tracking (the audited baseline).
3. Genuine Riemannian Armijo line search with sufficient increase.
4. Riemannian Barzilai-Borwein (RBB) gradient method.
5. Wen & Yin (2013) feasible Cayley transform method.
"""

import time
import numpy as np


def clean_covariance(S, scale, eps_reg=1e-10):
    """Symmetrize and add small regularizing jitter."""
    p = S.shape[-1]
    return 0.5 * (S + S.T) + eps_reg * scale * np.eye(p)


def objective(W, A, S, s2):
    """
    Macro effective information objective:
    f(W) = 0.5 * [ ln det(s2 * Am Am^T + Sm) - ln det(Sm) ]
    where Am = W A W^T, Sm = W S W^T.
    """
    Am = W @ A @ W.T
    Sm = W @ S @ W.T
    M = s2 * (Am @ Am.T) + Sm
    sign_m, logdet_m = np.linalg.slogdet(M)
    sign_s, logdet_s = np.linalg.slogdet(Sm)
    if sign_m <= 0 or sign_s <= 0:
        return -1e12
    return 0.5 * (logdet_m - logdet_s)


def euclidean_gradient(W, A, S, s2):
    """
    Exact Euclidean gradient of objective f(W) with respect to W (shape: q x p).
    """
    Am = W @ A @ W.T
    Sm = W @ S @ W.T
    M = s2 * (Am @ Am.T) + Sm
    G = np.linalg.inv(M)
    Sinv = np.linalg.inv(Sm)
    P = s2 * (G @ Am)          # df / dAm
    Q = 0.5 * (G - Sinv)       # df / dSm
    grad = (P @ W @ A.T + P.T @ W @ A) + (Q @ W @ S.T + Q.T @ W @ S)
    return grad


def project_tangent(W, grad_e):
    """
    Riemannian gradient projection onto the tangent space of St(q, p):
    grad_R = grad_e - W @ grad_e^T @ W  (for W of shape q x p, W W^T = I_q).
    """
    return grad_e - W @ grad_e.T @ W


def retract_qr(W):
    """
    QR-based metric retraction on the Stiefel manifold St(q, p).
    For W (q x p), factors W^T = Q R, returns (Q * signs)^T.
    """
    Qm, R = np.linalg.qr(W.T)
    diag = np.diag(R)
    signs = np.where(diag < 0.0, -1.0, 1.0)
    return (Qm * signs).T


def generate_initializations(p, q, n_restarts, Sigma_x=None, seed_base=42):
    """Generate deterministic initial guesses on St(q, p)."""
    inits = []
    if Sigma_x is not None:
        _, ev = np.linalg.eigh(0.5 * (Sigma_x + Sigma_x.T))
        inits.append(ev[:, -q:].T.copy())
    
    start_r = 1 if Sigma_x is not None else 0
    for r in range(start_r, n_restarts):
        rng = np.random.default_rng(seed_base + r * 1000 + q * 17)
        Qm, R = np.linalg.qr(rng.standard_normal((p, q)))
        signs = np.where(np.diag(R) < 0.0, -1.0, 1.0)
        inits.append((Qm * signs).T.copy())
    return inits


def optimize_fixed(A, S, s2, q, n_restarts=12, max_iter=100, lr=0.05,
                   Sigma_x=None, seed_base=42):
    """
    Audited baseline: fixed-step Riemannian gradient ascent without line search,
    stopping criteria, or best-iterate tracking.
    """
    p = A.shape[0]
    inits = generate_initializations(p, q, n_restarts, Sigma_x=Sigma_x, seed_base=seed_base)
    
    best_f = -np.inf
    best_W = None
    total_steps = 0
    objective_evals = 0
    gradient_evals = 0
    retraction_evals = 0
    start_time = time.perf_counter()

    for W in inits:
        for _ in range(max_iter):
            total_steps += 1
            ge = euclidean_gradient(W, A, S, s2)
            gradient_evals += 1
            gr = project_tangent(W, ge)
            W = retract_qr(W + lr * gr)
            retraction_evals += 1
        f_final = objective(W, A, S, s2)
        objective_evals += 1
        if f_final > best_f:
            best_f = f_final
            best_W = W

    elapsed = time.perf_counter() - start_time
    return {
        "best_f": best_f,
        "best_W": best_W,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "method": f"fixed_{n_restarts}_{max_iter}",
        "objective_evals": objective_evals,
        "gradient_evals": gradient_evals,
        "retraction_evals": retraction_evals,
    }


def optimize_monotone_backtracking(A, S, s2, q, n_restarts=12, max_iter=100, init_lr=0.05,
                                   Sigma_x=None, seed_base=42, tol=1e-8, max_ls=30):
    """
    Monotone Riemannian backtracking used in the audited production comparison.

    The acceptance rule requires strict improvement only.  It is intentionally
    kept separate from :func:`optimize_armijo`, which implements the genuine
    Armijo sufficient-increase condition.
    """
    p = A.shape[0]
    inits = generate_initializations(p, q, n_restarts, Sigma_x=Sigma_x, seed_base=seed_base)
    
    best_f = -np.inf
    best_W = None
    total_steps = 0
    objective_evals = 0
    gradient_evals = 0
    retraction_evals = 0
    start_time = time.perf_counter()

    for W in inits:
        f_cur = objective(W, A, S, s2)
        objective_evals += 1
        if f_cur > best_f:
            best_f = f_cur
            best_W = W.copy()
            
        step = init_lr
        for it in range(max_iter):
            total_steps += 1
            ge = euclidean_gradient(W, A, S, s2)
            gradient_evals += 1
            gr = project_tangent(W, ge)
            gnorm = np.linalg.norm(gr)
            if gnorm < tol:
                break
                
            accepted = False
            for _ in range(max_ls):
                W_cand = retract_qr(W + step * gr)
                f_cand = objective(W_cand, A, S, s2)
                retraction_evals += 1
                objective_evals += 1
                if f_cand > f_cur:
                    W = W_cand
                    f_cur = f_cand
                    accepted = True
                    step *= 1.5
                    if f_cur > best_f:
                        best_f = f_cur
                        best_W = W.copy()
                    break
                step *= 0.5
                
            if not accepted:
                break

    elapsed = time.perf_counter() - start_time
    return {
        "best_f": best_f,
        "best_W": best_W,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "method": f"monotone_backtracking_{n_restarts}_{max_iter}",
        "objective_evals": objective_evals,
        "gradient_evals": gradient_evals,
        "retraction_evals": retraction_evals,
    }


def optimize_armijo(A, S, s2, q, n_restarts=12, max_iter=100, init_lr=0.05,
                    Sigma_x=None, seed_base=42, tol=1e-8, max_ls=30,
                    c1=1e-4, tau=0.5):
    """Genuine maximization Armijo line search on the Stiefel manifold.

    A candidate is accepted only when it satisfies
    ``f(W_new) >= f(W) + c1 * step * ||grad f(W)||^2``.  This method is
    deliberately additive: the original monotone-backtracking benchmark is
    unchanged so its published numerical results remain reproducible.
    """
    p = A.shape[0]
    inits = generate_initializations(p, q, n_restarts, Sigma_x=Sigma_x, seed_base=seed_base)

    best_f = -np.inf
    best_W = None
    total_steps = 0
    objective_evals = 0
    gradient_evals = 0
    retraction_evals = 0
    start_time = time.perf_counter()

    for W in inits:
        f_cur = objective(W, A, S, s2)
        objective_evals += 1
        if f_cur > best_f:
            best_f = f_cur
            best_W = W.copy()

        step = init_lr
        for _ in range(max_iter):
            total_steps += 1
            ge = euclidean_gradient(W, A, S, s2)
            gradient_evals += 1
            gr = project_tangent(W, ge)
            gnorm_sq = float(np.sum(gr * gr))
            if gnorm_sq < tol ** 2:
                break

            accepted = False
            for _ in range(max_ls):
                W_cand = retract_qr(W + step * gr)
                f_cand = objective(W_cand, A, S, s2)
                retraction_evals += 1
                objective_evals += 1
                if f_cand >= f_cur + c1 * step * gnorm_sq:
                    W = W_cand
                    f_cur = f_cand
                    accepted = True
                    step /= tau
                    if f_cur > best_f:
                        best_f = f_cur
                        best_W = W.copy()
                    break
                step *= tau

            if not accepted:
                break

    elapsed = time.perf_counter() - start_time
    return {
        "best_f": best_f,
        "best_W": best_W,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "method": f"armijo_{n_restarts}_{max_iter}",
        "objective_evals": objective_evals,
        "gradient_evals": gradient_evals,
        "retraction_evals": retraction_evals,
    }


def optimize_rbb(A, S, s2, q, n_restarts=12, max_iter=100, init_lr=0.05,
                 Sigma_x=None, seed_base=42, tol=1e-8):
    """
    Riemannian Barzilai-Borwein (RBB) two-point step-size method.
    Uses vector transport via tangent space projection.
    """
    p = A.shape[0]
    inits = generate_initializations(p, q, n_restarts, Sigma_x=Sigma_x, seed_base=seed_base)
    
    best_f = -np.inf
    best_W = None
    total_steps = 0
    objective_evals = 0
    gradient_evals = 0
    retraction_evals = 0
    start_time = time.perf_counter()

    for W in inits:
        f_cur = objective(W, A, S, s2)
        objective_evals += 1
        if f_cur > best_f:
            best_f = f_cur
            best_W = W.copy()
            
        ge = euclidean_gradient(W, A, S, s2)
        gradient_evals += 1
        gr = project_tangent(W, ge)
        step = init_lr
        
        for it in range(max_iter):
            total_steps += 1
            gnorm = np.linalg.norm(gr)
            if gnorm < tol:
                break
                
            accepted = False
            for _ in range(25):
                W_cand = retract_qr(W + step * gr)
                f_cand = objective(W_cand, A, S, s2)
                retraction_evals += 1
                objective_evals += 1
                if f_cand > f_cur:
                    W_prev = W
                    gr_prev = gr
                    W = W_cand
                    f_cur = f_cand
                    accepted = True
                    if f_cur > best_f:
                        best_f = f_cur
                        best_W = W.copy()
                    break
                step *= 0.5
                
            if not accepted:
                break
                
            ge_new = euclidean_gradient(W, A, S, s2)
            gradient_evals += 1
            gr_new = project_tangent(W, ge_new)
            
            s_vec = project_tangent(W, W - W_prev)
            y_vec = gr_new - project_tangent(W, gr_prev)
            
            sy = np.sum(s_vec * y_vec)
            ss = np.sum(s_vec * s_vec)
            
            if abs(sy) > 1e-14 and ss > 1e-14:
                step = ss / abs(sy)
                step = float(np.clip(step, 1e-4, 1e4))
            else:
                step = init_lr
                
            gr = gr_new

    elapsed = time.perf_counter() - start_time
    return {
        "best_f": best_f,
        "best_W": best_W,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "method": f"rbb_{n_restarts}_{max_iter}",
        "objective_evals": objective_evals,
        "gradient_evals": gradient_evals,
        "retraction_evals": retraction_evals,
    }


def optimize_wen_yin(A, S, s2, q, n_restarts=12, max_iter=100, init_lr=0.05,
                     Sigma_x=None, seed_base=42, tol=1e-8):
    """
    Wen & Yin (2013) feasible method with Cayley transform retraction.
    """
    p = A.shape[0]
    inits = generate_initializations(p, q, n_restarts, Sigma_x=Sigma_x, seed_base=seed_base)
    
    best_f = -np.inf
    best_W = None
    total_steps = 0
    objective_evals = 0
    gradient_evals = 0
    retraction_evals = 0
    start_time = time.perf_counter()

    for W in inits:
        f_cur = objective(W, A, S, s2)
        objective_evals += 1
        if f_cur > best_f:
            best_f = f_cur
            best_W = W.copy()
            
        step = init_lr
        for it in range(max_iter):
            total_steps += 1
            ge = euclidean_gradient(W, A, S, s2)
            gradient_evals += 1
            gr = project_tangent(W, ge)
            if np.linalg.norm(gr) < tol:
                break
                
            U = ge.T @ W - W.T @ ge
            
            accepted = False
            for _ in range(25):
                Mat1 = np.eye(p) - 0.5 * step * U
                Mat2 = np.eye(p) + 0.5 * step * U
                W_cand_T = np.linalg.solve(Mat1, Mat2 @ W.T)
                W_cand = W_cand_T.T
                retraction_evals += 1

                f_cand = objective(W_cand, A, S, s2)
                objective_evals += 1
                if f_cand > f_cur:
                    W = W_cand
                    f_cur = f_cand
                    accepted = True
                    step *= 1.4
                    if f_cur > best_f:
                        best_f = f_cur
                        best_W = W.copy()
                    break
                step *= 0.5
                
            if not accepted:
                break

    elapsed = time.perf_counter() - start_time
    return {
        "best_f": best_f,
        "best_W": best_W,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "method": f"wen_yin_{n_restarts}_{max_iter}",
        "objective_evals": objective_evals,
        "gradient_evals": gradient_evals,
        "retraction_evals": retraction_evals,
    }


def compute_cefi_with_method(A, S, Sigma_x, method_fn, kappa=1.0, qs=None, tie_tol=1e-7, **kw):
    """
    Computes CEFI over macro dimensions q using the given manifold optimizer function.
    """
    p = A.shape[0]
    scale = max(np.trace(Sigma_x) / p, 1e-12)
    s2 = (kappa ** 2) * scale
    Sc = clean_covariance(S, scale)
    
    # Micro EI
    M = s2 * (A @ A.T) + Sc
    eim = 0.5 * (np.linalg.slogdet(M)[1] - np.linalg.slogdet(Sc)[1])
    dens = eim / p
    
    if qs is None:
        qs = list(range(1, p))
        
    spec = {}
    details = {}
    for q in qs:
        res = method_fn(A, Sc, s2, q, Sigma_x=Sigma_x, **kw)
        f = res["best_f"]
        spec[q] = f / q - dens
        details[q] = res
        
    mx = max(spec.values())
    qstar = min(q for q, v in spec.items() if v >= mx - tie_tol)
    return spec[qstar], qstar, spec, eim, details
