"""
SVD-Based Causal Emergence Engine (Liu, Pan, Wang, Yang, Yuan, Zhang, Physical Review E, 2025)
==================================================================================
Implements exact singular value decomposition (SVD) causal emergence for linear
stochastic Gaussian systems without iterative numerical search.
"""

from typing import Tuple, Dict, Optional
import numpy as np


def compute_svd_causal_emergence(
    A: np.ndarray,
    Sigma_eps: np.ndarray,
    Sigma_x: Optional[np.ndarray] = None,
    kappa_do: float = 1.0,
    eps_reg: float = 1e-10
) -> Tuple[float, int, np.ndarray, Dict[int, float]]:
    """
    Computes analytical SVD-based Causal Emergence (Yang et al., Phys. Rev. E, 2025).

    The transfer operator relative to noise is K = L_eps^{-1} A, with Sigma_eps = L_eps L_eps^T.
    Singular values s_1 >= s_2 >= ... >= s_p determine the exact upper bound of macro EI.

    Parameters
    ----------
    A : np.ndarray of shape (p, p)
    Sigma_eps : np.ndarray of shape (p, p)
    Sigma_x : Optional[np.ndarray] of shape (p, p)
    kappa_do : float (default=1.0)
    eps_reg : float

    Returns
    -------
    ce_svd : float
        Maximal SVD-based causal emergence density.
    q_star_svd : int
        Optimal macro dimension derived from singular spectrum.
    singular_values : np.ndarray
        Array of singular values s_1 >= ... >= s_p.
    density_deltas : Dict[int, float]
        Emergence density values for each q in 1..p-1.
    """
    p = A.shape[0]

    # Scale-invariant intervention variance
    if Sigma_x is not None:
        var_scale = float(np.trace(Sigma_x) / float(p))
    else:
        var_scale = float(np.trace(Sigma_eps) / float(p))

    var_scale = max(var_scale, 1e-12)
    eff_sigma_do_sq = (kappa_do ** 2) * var_scale

    # Regularized Cholesky of noise covariance
    Sigma_clean = 0.5 * (Sigma_eps + Sigma_eps.T) + eps_reg * var_scale * np.eye(p)
    L = np.linalg.cholesky(Sigma_clean)

    # Normalized transfer matrix K = L^{-1} A
    K = np.linalg.solve(L, A)

    # Singular value decomposition
    _, s, _ = np.linalg.svd(K, full_matrices=False)
    # s is sorted descending: s_1 >= s_2 >= ... >= s_p

    # Full micro EI from singular spectrum
    ei_micro_terms = np.log(1.0 + eff_sigma_do_sq * (s ** 2))
    ei_micro = float(0.5 * np.sum(ei_micro_terms))
    ei_micro_density = ei_micro / float(p)

    # Macro EI for top q singular modes
    density_deltas = {}
    for q in range(1, p):
        ei_q_svd = float(0.5 * np.sum(ei_micro_terms[:q]))
        delta_q = (ei_q_svd / float(q)) - ei_micro_density
        density_deltas[q] = delta_q

    best_q_svd = max(density_deltas, key=density_deltas.get)
    ce_svd = float(density_deltas[best_q_svd])

    return ce_svd, best_q_svd, s, density_deltas
