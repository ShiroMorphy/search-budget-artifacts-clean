"""Compatibility API for the canonical deterministic Stiefel optimizer.

All production and legacy callers are routed through ``evaluate_batch_cefi`` so
there is one mathematical estimator, one initialization scheme, and one set of
numerical conventions on CPU and CUDA.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - only used in minimal installations
    torch = None


def initialize_stiefel_matrix(
    p: int,
    q: int,
    method: str = "pca",
    cov_matrix: Optional[np.ndarray] = None,
    seed: int = 42,
) -> np.ndarray:
    """Return a deterministic row-orthonormal matrix in V_q(R^p)."""
    if not 1 <= q <= p:
        raise ValueError(f"q={q} must satisfy 1 <= q <= p={p}")
    if method == "pca" and cov_matrix is not None:
        covariance = 0.5 * (np.asarray(cov_matrix) + np.asarray(cov_matrix).T)
        _, eigvecs = np.linalg.eigh(covariance)
        W = eigvecs[:, -q:].T
    else:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((p, q))
        Q, R = np.linalg.qr(raw)
        signs = np.where(np.diag(R) < 0.0, -1.0, 1.0)
        W = (Q * signs).T
    return np.asarray(W, dtype=np.float64)


def optimize_stiefel_torch(
    A_np: np.ndarray,
    Sigma_eps_np: np.ndarray,
    q: int,
    Sigma_x_np: Optional[np.ndarray] = None,
    kappa_do: float = 1.0,
    n_restarts: int = 12,
    max_iter: int = 100,
    lr: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """Optimize one macro dimension with the shared canonical CPU engine."""
    if torch is None:
        raise RuntimeError("PyTorch is required for Stiefel optimization")

    from .cuda_stiefel import evaluate_batch_cefi

    covariance = Sigma_eps_np if Sigma_x_np is None else Sigma_x_np
    result = evaluate_batch_cefi(
        np.asarray(A_np, dtype=np.float64)[None, ...],
        np.asarray(Sigma_eps_np, dtype=np.float64)[None, ...],
        np.asarray(covariance, dtype=np.float64)[None, ...],
        [q],
        n_restarts=n_restarts,
        max_iter=max_iter,
        kappa=kappa_do,
        learning_rate=lr,
        device=torch.device("cpu"),
        search_dtype=torch.float64,
        return_best_w=True,
    )
    return result[4][0], float(result[3][0])


def optimize_coarse_graining_stiefel(
    A: np.ndarray,
    Sigma_eps: np.ndarray,
    q: int,
    Sigma_x: Optional[np.ndarray] = None,
    kappa_do: float = 1.0,
    n_restarts: int = 12,
    max_iter: int = 100,
    lr: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """Maximize canonical macro EI over V_q(R^p), deterministically."""
    return optimize_stiefel_torch(
        A,
        Sigma_eps,
        q=q,
        Sigma_x_np=Sigma_x,
        kappa_do=kappa_do,
        n_restarts=n_restarts,
        max_iter=max_iter,
        lr=lr,
    )
