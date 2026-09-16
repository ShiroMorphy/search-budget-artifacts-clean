"""
Causal Emergence for Financial Markets
======================================
Analytical and neural multiscale information-theoretic measures of systemic risk.
"""

from .analytical_ei import (
    compute_continuous_ei,
    compute_macro_dynamics,
    compute_macro_ei,
    compute_emergence_spectrum
)
from .micro_var import fit_micro_var1
from .stiefel_optimizer import optimize_coarse_graining_stiefel

__all__ = [
    "compute_continuous_ei",
    "compute_macro_dynamics",
    "compute_macro_ei",
    "compute_emergence_spectrum",
    "fit_micro_var1",
    "optimize_coarse_graining_stiefel",
]
