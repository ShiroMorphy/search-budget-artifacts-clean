"""Deterministic batched Stiefel optimization for the canonical CEFI estimator.

This module is the single GPU/CPU implementation of the estimator described in
the manuscript: constructed macro channels ``W A W'``, isotropic Gaussian
interventions scaled by the micro-state average variance, and EI measured in
nats.  Deterministic initial bases are shared across batch partitions so that a
system receives exactly the same estimate whether evaluated alone or in a batch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch


ESTIMATOR_SPEC = "isotropic_constructed_macro_nats_full_precision_v3"
FP64_REFINEMENT_STEPS = 0


def estimator_fingerprint() -> str:
    """SHA-256 of the exact canonical estimator source used by a run."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _select_q_with_global_tolerance(
    cefi_spectrum: torch.Tensor,
    q_values: Iterable[int],
    tie_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return selected column and q using the global epsilon tie rule."""
    q_values = list(q_values)
    global_max = cefi_spectrum.max(dim=1).values
    eligible = cefi_spectrum >= global_max[:, None] - float(tie_tolerance)
    selected_idx = eligible.to(torch.int64).argmax(dim=1)
    q_tensor = torch.as_tensor(q_values, dtype=torch.long, device=cefi_spectrum.device)
    return selected_idx, q_tensor[selected_idx]


def _clean_covariance(S: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetrize a covariance matrix and add scale-relative jitter."""
    p = S.shape[-1]
    eye = torch.eye(p, dtype=S.dtype, device=S.device)
    return 0.5 * (S + S.transpose(-2, -1)) + 1e-10 * scale[..., None, None] * eye


def evaluate_batch_cefi(
    A_np: np.ndarray,
    Sigma_eps_np: np.ndarray,
    Sigma_x_np: np.ndarray,
    q_candidates: Iterable[int],
    *,
    n_restarts: int = 12,
    max_iter: int = 100,
    kappa: float = 1.0,
    learning_rate: float = 0.05,
    tie_tolerance: float = 1e-7,
    device: Optional[torch.device] = None,
    search_dtype: torch.dtype = torch.float64,
    fp64_refinement_steps: int = FP64_REFINEMENT_STEPS,
    return_best_w: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Evaluate canonical CEFI for a batch of fitted linear systems.

    The intervention variance for each system is
    ``kappa**2 * trace(Sigma_x) / p`` and is used unchanged at every macro
    dimension.  This is the isotropic, scale-invariant intervention stated in
    the manuscript.  All logarithms are natural, hence EI and CEFI are in nats.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A64 = torch.as_tensor(A_np, dtype=torch.float64, device=device)
    Sigma_eps64 = torch.as_tensor(Sigma_eps_np, dtype=torch.float64, device=device)
    Sigma_x64 = torch.as_tensor(Sigma_x_np, dtype=torch.float64, device=device)
    if A64.ndim != 3 or Sigma_eps64.shape != A64.shape or Sigma_x64.shape != A64.shape:
        raise ValueError("A, Sigma_eps, and Sigma_x must all have shape (B, p, p)")

    batch_size, p, _ = A64.shape
    state_scale64 = torch.diagonal(Sigma_x64, dim1=-2, dim2=-1).sum(-1) / float(p)
    state_scale64 = torch.clamp(state_scale64, min=1e-12)
    sigma_do_sq64 = (float(kappa) ** 2) * state_scale64
    Sigma_eps64 = _clean_covariance(Sigma_eps64, state_scale64)

    micro_eff = sigma_do_sq64[:, None, None] * (A64 @ A64.transpose(-2, -1)) + Sigma_eps64
    ei_micro = 0.5 * (
        torch.linalg.slogdet(micro_eff)[1] - torch.linalg.slogdet(Sigma_eps64)[1]
    )
    micro_density = ei_micro / float(p)

    # The search can run in FP32 on consumer GPUs; every objective used for
    # restart selection and reporting is recomputed below in FP64.
    A = A64.to(search_dtype)
    Sigma_eps = Sigma_eps64.to(search_dtype)
    Sigma_x = Sigma_x64.to(search_dtype)
    sigma_do_sq = sigma_do_sq64.to(search_dtype)

    q_values = sorted(int(value) for value in q_candidates)
    if return_best_w and len(q_values) != 1:
        raise ValueError("return_best_w=True requires exactly one q candidate")

    # Build every initialization on the host and then transfer it.  This keeps
    # CPU and CUDA searches on exactly the same starting subspaces; device-local
    # RNGs and eigensolvers are otherwise allowed to choose different bases.
    sigma_x_host = np.asarray(Sigma_x_np, dtype=np.float64)
    sigma_x_host = 0.5 * (sigma_x_host + np.swapaxes(sigma_x_host, -2, -1))
    _, eigvecs_host = np.linalg.eigh(sigma_x_host)

    cefi_by_q = []
    macro_ei_by_q = []
    best_w_by_system = None

    for q in q_values:
        if q < 1 or q >= p:
            raise ValueError(f"q={q} must satisfy 1 <= q < p={p}")

        # Restart zero is system-specific PCA. Remaining restarts are fixed
        # orthogonal bases depending only on (q, restart), not batch position.
        pca_basis = np.swapaxes(eigvecs_host[:, :, -q:], -2, -1).copy()
        initial = [torch.as_tensor(pca_basis, dtype=search_dtype, device=device)]
        for restart in range(1, n_restarts):
            rng = np.random.default_rng(42 + restart * 1000 + q * 17)
            raw = rng.standard_normal((p, q))
            Q_host, R_host = np.linalg.qr(raw)
            signs = np.where(np.diag(R_host) < 0.0, -1.0, 1.0)
            basis_host = (Q_host * signs).T.copy()
            basis = torch.as_tensor(basis_host, dtype=search_dtype, device=device)
            initial.append(basis.unsqueeze(0).expand(batch_size, -1, -1))

        # Shape: (restart, batch, q, p). Vectorizing restarts is materially
        # faster for the many small matrices in this application.
        W = torch.stack(initial, dim=0).clone().requires_grad_(True)
        A_r = A.unsqueeze(0)
        S_r = Sigma_eps.unsqueeze(0)
        sigma_r = sigma_do_sq[None, :, None, None]

        refinement_steps = (
            min(int(fp64_refinement_steps), max_iter)
            if search_dtype == torch.float32
            else 0
        )
        search_steps = max_iter - refinement_steps
        for _ in range(search_steps):
            A_macro = W @ A_r @ W.transpose(-2, -1)
            S_macro = W @ S_r @ W.transpose(-2, -1)
            macro_eff = sigma_r * (A_macro @ A_macro.transpose(-2, -1)) + S_macro
            objective = 0.5 * (
                torch.linalg.slogdet(macro_eff)[1] - torch.linalg.slogdet(S_macro)[1]
            )
            grad = torch.autograd.grad(objective.sum(), W)[0]

            with torch.no_grad():
                grad_r = grad - W @ grad.transpose(-2, -1) @ W
                candidate = W + float(learning_rate) * grad_r
                Q, R = torch.linalg.qr(candidate.transpose(-2, -1))
                diag = torch.diagonal(R, dim1=-2, dim2=-1)
                signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
                retracted = (Q * signs.unsqueeze(-2)).transpose(-2, -1)
            W = retracted.clone().requires_grad_(True)

        # Finish a mixed-precision search with FP64 manifold steps. The total
        # iteration budget remains exactly max_iter (50 FP32 + 50 FP64 for the
        # production 100-step configuration).
        if refinement_steps:
            W = W.to(torch.float64).detach().requires_grad_(True)
            A_ref = A64.unsqueeze(0)
            S_ref = Sigma_eps64.unsqueeze(0)
            sigma_ref = sigma_do_sq64[None, :, None, None]
            for _ in range(refinement_steps):
                A_macro = W @ A_ref @ W.transpose(-2, -1)
                S_macro = W @ S_ref @ W.transpose(-2, -1)
                macro_eff = sigma_ref * (A_macro @ A_macro.transpose(-2, -1)) + S_macro
                objective = 0.5 * (
                    torch.linalg.slogdet(macro_eff)[1] - torch.linalg.slogdet(S_macro)[1]
                )
                grad = torch.autograd.grad(objective.sum(), W)[0]
                with torch.no_grad():
                    grad_r = grad - W @ grad.transpose(-2, -1) @ W
                    candidate = W + float(learning_rate) * grad_r
                    Q, R = torch.linalg.qr(candidate.transpose(-2, -1))
                    diag = torch.diagonal(R, dim1=-2, dim2=-1)
                    signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
                    W_next = (Q * signs.unsqueeze(-2)).transpose(-2, -1)
                W = W_next.clone().requires_grad_(True)

        with torch.no_grad():
            W64 = W.to(torch.float64)
            A_macro = W64 @ A64.unsqueeze(0) @ W64.transpose(-2, -1)
            S_macro = W64 @ Sigma_eps64.unsqueeze(0) @ W64.transpose(-2, -1)
            macro_eff = sigma_do_sq64[None, :, None, None] * (A_macro @ A_macro.transpose(-2, -1)) + S_macro
            objectives = 0.5 * (
                torch.linalg.slogdet(macro_eff)[1] - torch.linalg.slogdet(S_macro)[1]
            )
            best_obj_q, restart_idx = objectives.max(dim=0)
            cefi_q = best_obj_q / float(q) - micro_density

            cefi_by_q.append(cefi_q)
            macro_ei_by_q.append(best_obj_q)

            if return_best_w:
                batch_idx = torch.arange(batch_size, device=device)
                selected_w = W64[restart_idx, batch_idx]
                best_w_by_system = selected_w.clone()

    # Apply the manuscript's tie rule against the *global* maximum: among all
    # dimensions within epsilon of max_q CEFI(q), select the smallest q.
    cefi_spectrum = torch.stack(cefi_by_q, dim=1)
    macro_spectrum = torch.stack(macro_ei_by_q, dim=1)
    selected_idx, best_q = _select_q_with_global_tolerance(
        cefi_spectrum, q_values, tie_tolerance
    )
    batch_idx = torch.arange(batch_size, device=device)
    best_cefi = cefi_spectrum[batch_idx, selected_idx]
    best_macro_ei = macro_spectrum[batch_idx, selected_idx]

    best_w_np = None if best_w_by_system is None else best_w_by_system.cpu().numpy()
    return (
        best_cefi.cpu().numpy(),
        best_q.cpu().numpy(),
        ei_micro.cpu().numpy(),
        best_macro_ei.cpu().numpy(),
        best_w_np,
    )
