"""
Micro VAR(1) Estimation Engine
==============================
Estimates continuous vector autoregressive models and covariance structures
with statistical shrinkage and regularization for high-dimensional financial series.
"""

from typing import Tuple, Optional
import numpy as np


def ledoit_wolf_shrinkage(residuals: np.ndarray) -> np.ndarray:
    """
    Pure NumPy vectorized implementation of the Ledoit & Wolf (2004) analytical shrinkage covariance estimator:
        Sigma = (1 - delta) * S + delta * mu * I_p
    """
    T, p = residuals.shape
    # Sample covariance (unbiased)
    S = (residuals.T @ residuals) / T
    # Target: mu * I_p where mu = tr(S) / p
    mu = np.trace(S) / p
    target = mu * np.eye(p)

    # Calculate shrinkage intensity delta
    # d2 = ||S - mu*I||_F^2
    d2 = np.sum((S - target) ** 2)

    # Vectorized asymptotic variance: b_bar^2 = (1 / T^2) * sum_t ||x_t x_t.T - S||_F^2
    X2 = residuals ** 2
    sum_outer_sq = X2.T @ X2
    b_bar_sq = float((np.sum(sum_outer_sq) - T * np.sum(S ** 2)) / (T ** 2))

    # Optimal shrinkage intensity bounded in [0, 1]
    b2 = min(b_bar_sq, d2)
    delta = 0.0 if d2 <= 0 else max(0.0, min(1.0, b2 / d2))

    shrunk_cov = (1.0 - delta) * S + delta * target
    return 0.5 * (shrunk_cov + shrunk_cov.T)


def fit_micro_var1(
    returns_matrix: np.ndarray,
    method: str = "ledoit_wolf",
    ridge_alpha: float = 1e-4,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a VAR(1) system: x_{t+1} = A x_t + eps_{t+1}

    Parameters
    ----------
    returns_matrix : np.ndarray of shape (T, p)
        Time series of returns over a window of length T with p assets/sectors.
    method : str, optional (default='ledoit_wolf')
        Covariance estimation technique: 'ledoit_wolf', 'ridge', or 'ols'.
    ridge_alpha : float, optional (default=1e-4)
        L2 regularization parameter for transition matrix estimation.
    center : bool, optional (default=True)
        Whether to demean the time series prior to estimation.

    Returns
    -------
    A : np.ndarray of shape (p, p)
        Estimated transition matrix.
    Sigma_eps : np.ndarray of shape (p, p)
        Estimated residual covariance matrix.
    """
    T, p = returns_matrix.shape
    if T < p + 2:
        raise ValueError(f"Window length T={T} is too short for dimension p={p}. Require T >= p + 2.")

    X = returns_matrix.copy()
    if center:
        X = X - np.mean(X, axis=0, keepdims=True)

    # Construct lagged matrices
    X_lag = X[:-1, :]  # Shape: (T-1, p)
    X_lead = X[1:, :]  # Shape: (T-1, p)

    # Estimate transition matrix A via Scale-Invariant Ridge / OLS
    XtX = X_lag.T @ X_lag
    XtY = X_lag.T @ X_lead
    xtx_scale = np.trace(XtX) / float(p)

    if ridge_alpha > 0 and xtx_scale > 0:
        eff_ridge = ridge_alpha * xtx_scale * np.eye(p)
        A_T = np.linalg.solve(XtX + eff_ridge, XtY)
    else:
        A_T = np.linalg.lstsq(X_lag, X_lead, rcond=None)[0]

    A = A_T.T  # Shape: (p, p)


    # Compute innovations / residuals
    residuals = X_lead - X_lag @ A_T  # Shape: (T-1, p)

    # Estimate residual covariance Sigma_eps
    if method == "ledoit_wolf":
        Sigma_eps = ledoit_wolf_shrinkage(residuals)
    elif method == "ridge":
        sample_cov = (residuals.T @ residuals) / (T - 1)
        Sigma_eps = sample_cov + ridge_alpha * np.eye(p)
    else:  # OLS sample covariance
        Sigma_eps = (residuals.T @ residuals) / (T - 1 - p)

    # Guarantee symmetry
    Sigma_eps = 0.5 * (Sigma_eps + Sigma_eps.T)

    return A, Sigma_eps

