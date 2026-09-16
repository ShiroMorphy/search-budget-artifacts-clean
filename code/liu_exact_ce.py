"""
Exact Causal Emergence Formulation from Liu, Yuan & Zhang (2024)
================================================================
Implements the exact dimension-averaged Effective Information Delta J for linear
stochastic iteration systems under bounded uniform interventions do(x_t ~ U([-L/2, L/2]^p)).
"""

from typing import Tuple, Dict, Optional
import numpy as np


def compute_liu_exact_emergence(
    A: np.ndarray,
    Sigma_eps: np.ndarray,
    W_dict: Dict[int, np.ndarray],
    L_scale: Optional[float] = None,
    Sigma_x: Optional[np.ndarray] = None,
    eps_reg: float = 1e-10
) -> Tuple[float, int, Dict[int, float]]:
    """
    Computes exact dimension-averaged Causal Emergence Delta J (Liu et al., 2024):
        J(x) = (1/p) * 0.5 * ln det(I_p + (L^2 / 12) * A A^T Sigma_eps^{-1})
        J(y) = (1/q) * 0.5 * ln det(I_q + (L^2 / 12) * A_M A_M^T Sigma_M^{-1})
        Delta J = max_{q < p} [ J(y) - J(x) ]
        with A_M = W A W^T,  Sigma_M = W Sigma_eps W^T

    Parameters
    ----------
    A : np.ndarray of shape (p, p)
    Sigma_eps : np.ndarray of shape (p, p)
    W_dict : Dict[int, np.ndarray]
        Dictionary of coarse-graining projection matrices W of shape (q, p) for each q.
    L_scale : Optional[float]
        Domain size L. If None, set to sqrt(12 * var_scale) for scale invariance.
    Sigma_x : Optional[np.ndarray]

    Returns
    -------
    delta_J_max : float
        Maximum dimension-averaged causal emergence Delta J.
    q_star_liu : int
        Optimal macro dimension.
    delta_J_spectrum : Dict[int, float]
    """
    p = A.shape[0]
    if Sigma_x is not None:
        var_scale = float(np.trace(Sigma_x) / float(p))
    else:
        var_scale = float(np.trace(Sigma_eps) / float(p))

    var_scale = max(var_scale, 1e-12)

    # Variance of U([-L/2, L/2]) is L^2 / 12
    if L_scale is None:
        sigma_unif_sq = var_scale
    else:
        sigma_unif_sq = (L_scale ** 2) / 12.0

    # Micro J
    Sigma_clean = 0.5 * (Sigma_eps + Sigma_eps.T) + eps_reg * var_scale * np.eye(p)
    L_mat = np.linalg.cholesky(Sigma_clean)
    Y = np.linalg.solve(L_mat, A)
    M = np.eye(p) + sigma_unif_sq * (Y @ Y.T)
    j_micro = float(0.5 * np.linalg.slogdet(M)[1] / float(p))

    # Macro J(q)
    delta_J_dict = {}
    for q, W in W_dict.items():
        A_M = W @ A @ W.T
        Sigma_M = W @ Sigma_clean @ W.T
        Sigma_M = 0.5 * (Sigma_M + Sigma_M.T) + eps_reg * var_scale * np.eye(q)
        
        L_M = np.linalg.cholesky(Sigma_M)
        Y_M = np.linalg.solve(L_M, A_M)
        M_macro = np.eye(q) + sigma_unif_sq * (Y_M @ Y_M.T)
        j_macro = float(0.5 * np.linalg.slogdet(M_macro)[1] / float(q))
        
        delta_J_dict[q] = j_macro - j_micro

    best_q = max(delta_J_dict, key=delta_J_dict.get)
    delta_J_max = float(delta_J_dict[best_q])

    return delta_J_max, best_q, delta_J_dict
