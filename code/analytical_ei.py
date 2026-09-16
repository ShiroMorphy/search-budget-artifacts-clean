"""
Analytical Continuous Effective Information (EI) Engine
======================================================
Implements exact, scale-invariant information-theoretic formulations for linear
continuous-state Gaussian Markov systems (VAR(1)) under maximum entropy intervention.
"""

from typing import Dict, Tuple, Optional
import numpy as np


def compute_continuous_ei(
    A: np.ndarray,
    Sigma_eps: np.ndarray,
    Sigma_x: Optional[np.ndarray] = None,
    kappa_do: float = 1.0,
    sigma_do: Optional[float] = None,
    eps_regularization: float = 1e-12
) -> float:
    """
    Computes exact, scale-invariant Effective Information (EI) for a linear Gaussian continuous system:
        x_{t+1} = A x_t + eps_{t+1},   eps_t ~ N(0, Sigma_eps)

    Under an intervention do(x_t ~ N(0, Sigma_do)), where Sigma_do = sigma_do^2 * I_p.
    To ensure strict invariance under rescaling of returns (e.g. decimal vs percentage vs basis points):
        sigma_do^2 = kappa_do^2 * Tr(Sigma_x) / p

    Parameters
    ----------
    A : np.ndarray of shape (p, p)
        Transition matrix.
    Sigma_eps : np.ndarray of shape (p, p)
        Noise innovation covariance matrix.
    Sigma_x : Optional[np.ndarray] of shape (p, p)
        State covariance matrix of returns. If provided, ensures scale-invariance.
    kappa_do : float, optional (default=1.0)
        Dimensionless intervention strength relative to system total variance.
    sigma_do : Optional[float], optional
        Direct intervention scale (if provided, overrides scale-invariant calculation).
    eps_regularization : float, optional (default=1e-12)
        Small diagonal jitter for numerical stability.

    Returns
    -------
    ei : float
        Effective Information in nats.
    """
    p = A.shape[0]

    # Scale-invariant intervention scale
    if Sigma_x is not None:
        var_scale = float(np.trace(Sigma_x) / float(p))
    else:
        var_scale = float(np.trace(Sigma_eps) / float(p))

    var_scale = max(var_scale, 1e-12)

    if sigma_do is None:
        eff_sigma_do_sq = (kappa_do ** 2) * var_scale
    else:
        eff_sigma_do_sq = sigma_do ** 2

    # Symmetrize and regularize covariance relative to scale
    jitter = eps_regularization * var_scale * np.eye(p)
    Sigma_clean = 0.5 * (Sigma_eps + Sigma_eps.T) + jitter


    try:
        # Numerically stable Cholesky solve: L @ L.T = Sigma_clean
        L = np.linalg.cholesky(Sigma_clean)
        # Solve L @ Y = A  =>  Y = L^{-1} A
        Y = np.linalg.solve(L, A)
        # Construct M = I_p + eff_sigma_do_sq * (Y @ Y.T)
        M = np.eye(p) + eff_sigma_do_sq * (Y @ Y.T)
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            return 0.0
        return float(0.5 * logdet)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        Sigma_inv = np.linalg.pinv(Sigma_clean)
        M = np.eye(p) + eff_sigma_do_sq * (A @ A.T @ Sigma_inv)
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            return 0.0
        return float(0.5 * logdet)


def compute_macro_dynamics(
    A: np.ndarray,
    Sigma_eps: np.ndarray,
    W: np.ndarray,
    Sigma_x: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Projects micro-dynamics to macro-dynamics via coarse-graining matrix W:
        y_t = W x_t,   with W in Stiefel manifold V_q(R^p) (i.e. W @ W.T = I_q)

    Constructed macro interventional channel under the canonical lifting x = W^T y:
        y_{t+1} = W A W^T y_t + W eps_{t+1}
        A_y = W A W^T
        Sigma_eps_y = W Sigma_eps W^T
        Sigma_x_y = W Sigma_x W^T

    Parameters
    ----------
    A : np.ndarray of shape (p, p)
    Sigma_eps : np.ndarray of shape (p, p)
    W : np.ndarray of shape (q, p)
    Sigma_x : Optional[np.ndarray] of shape (p, p)

    Returns
    -------
    A_y : np.ndarray of shape (q, q)
    Sigma_eps_y : np.ndarray of shape (q, q)
    Sigma_x_y : Optional[np.ndarray] of shape (q, q)
    """
    q = W.shape[0]
    Sigma_eps_y = W @ Sigma_eps @ W.T

    A_y = W @ A @ W.T
    if Sigma_x is not None:
        Sigma_x_clean = 0.5 * (Sigma_x + Sigma_x.T) + 1e-12 * np.eye(Sigma_x.shape[0])
        Sigma_x_y = W @ Sigma_x_clean @ W.T
    else:
        Sigma_x_y = None

    return A_y, Sigma_eps_y, Sigma_x_y


def compute_macro_ei(
    A: np.ndarray,
    Sigma_eps: np.ndarray,
    W: np.ndarray,
    Sigma_x: Optional[np.ndarray] = None,
    kappa_do: float = 1.0,
    sigma_do: Optional[float] = None
) -> float:
    """
    Computes Effective Information for the macro-system induced by coarse-graining W.
    """
    A_y, Sigma_eps_y, _ = compute_macro_dynamics(A, Sigma_eps, W, Sigma_x=Sigma_x)
    if sigma_do is None and Sigma_x is not None:
        # Keep the micro-system's isotropic intervention variance unchanged at
        # every macro dimension, as specified in the manuscript.
        sigma_do = kappa_do * np.sqrt(max(float(np.trace(Sigma_x) / A.shape[0]), 1e-12))
    return compute_continuous_ei(
        A_y, Sigma_eps_y, Sigma_x=None, kappa_do=kappa_do, sigma_do=sigma_do
    )


def compute_emergence_spectrum(
    ei_micro: float,
    macro_ei_dict: Dict[int, float],
    p_micro: Optional[int] = None,
    tie_tolerance: float = 1e-7,
) -> Tuple[float, int, Dict[int, float], float]:
    """
    Evaluates Causal Emergence Financial Index (CEFI) and Causal Effective Dimension (q*).

    Parameters
    ----------
    ei_micro : float
        EI of the full micro system (dimension p).
    macro_ei_dict : Dict[int, float]
        Dictionary mapping dimension q -> optimal macro EI (EI_q^*).
    p_micro : Optional[int]
        Micro dimension p (if None, inferred from max(macro_ei_dict) + 1 or default).

    Returns
    -------
    cefi_density : float
        CEFI = max_{q < p} [ EI_q^* / q - EI_p / p ].
    q_star : int
        Dimension q* that maximizes macro EI density advantage.
    delta_ei_dict : Dict[int, float]
        Difference (EI_q^* / q - EI_p / p) for each candidate macro dimension q.
    cefi_raw : float
        Unnormalized difference max_{q < p} [ EI_q^* - EI_p ].
    """
    if p_micro is None:
        p_micro = max(macro_ei_dict.keys()) + 1

    ei_micro_density = ei_micro / float(p_micro)

    density_deltas = {
        q: (macro_ei / float(q) - ei_micro_density)
        for q, macro_ei in macro_ei_dict.items()
    }

    # Manuscript rule: choose the smallest q within epsilon of the global
    # maximum. A streaming comparison against the current best is not
    # equivalent when several near-ties form a chain.
    maximum_density = max(density_deltas.values())
    best_q = min(
        q for q, density in density_deltas.items()
        if density >= maximum_density - float(tie_tolerance)
    )
    cefi_density = float(density_deltas[best_q])

    # Raw unnormalized difference
    best_raw_q = max(macro_ei_dict, key=macro_ei_dict.get)
    cefi_raw = float(macro_ei_dict[best_raw_q] - ei_micro)

    return cefi_density, best_q, density_deltas, cefi_raw
