# Lean 4 formalization

Lean: `leanprover/lean4:v4.34.0-rc2`
Mathlib: commit `cf6bde11431e913de1d8804126d8006a7fb187c4`

This directory contains machine-checked proofs of two statements from the paper.
Every theorem listed below compiles; there are no `sorry`s and no unproved
assumptions. All statements are generic in the base ring and in the dimensions
`p` and `q`.

## Contents

### `CausalEmergence/Closure.lean`

| Theorem | Statement | Paper |
| :--- | :--- | :--- |
| `projected_step_decomposition` | `W *ᵥ (A *ᵥ x) = (W * A * Wᵀ) *ᵥ (W *ᵥ x) + (W * A * (1 - Wᵀ * W)) *ᵥ x` | Eq. (5), the closure defect |
| `autonomous_iff_closure` | The projected state is an autonomous linear system for every initial condition **if and only if** `W * A * (1 - Wᵀ * W) = 0` | Establishes that Eq. (4) defines a Galerkin-type surrogate rather than an exact marginal dynamics |
| `closure_of_left_invariant` | If `W * A = C * W` for some `C` and `W * Wᵀ = 1`, the residual vanishes | Sufficient condition for closure: invariance of the projected subspace |
| `transpose_compression` | `(W * S * Wᵀ)ᵀ = W * Sᵀ * Wᵀ` | Compressions preserve symmetry |
| `compression_smul` | `W * (c ^ 2 • S) * Wᵀ = c ^ 2 • (W * S * Wᵀ)` | Algebraic content of invariance under a common scalar rescaling |

`autonomous_iff_closure` is stronger than the statement made in the paper, which
asserts only one direction.

### `CausalEmergence/Compression.lean`

| Theorem | Statement | Paper |
| :--- | :--- | :--- |
| `compression_quadratic_lower_bound` | If `W * Wᵀ = 1` and `c * (v ⬝ᵥ v) ≤ v ⬝ᵥ (S *ᵥ v)` for all `v`, then `c * (u ⬝ᵥ u) ≤ u ⬝ᵥ ((W * S * Wᵀ) *ᵥ u)` for all `u` | Related compression inequality; not used in the current Proposition 1 proof |

Taking `c` to be the least eigenvalue of the innovation covariance gives the
familiar bound `λ_min (W Σ_ε Wᵀ) ≥ λ_min Σ_ε`. Stating it in the quadratic-form order rather
than through eigenvalues is more general and requires no spectral theory: it
holds over any commutative ring equipped with an order relation.

## Scope

The full statement of Proposition 1 also requires singular-value contraction
under orthonormal compression, `s_i(W A Wᵀ) ≤ s_i(A)`, which in turn reduces to
submultiplicativity, `s_i(X Y) ≤ ‖X‖₂ s_i(Y)`. That half is **not** formalized
here.

Mathlib provides singular values for linear maps between inner product spaces,
including the fact that they are antitone, together with the Hermitian spectral
theorem, `PosSemidef` and `PosDef` with their eigenvalue characterizations, and
square roots via the continuous functional calculus. It does not currently
provide a minimax or Courant-Fischer characterization, submultiplicativity of
singular values, or a matrix-level Rayleigh-Ritz bound, and it defines singular
values for linear maps rather than for matrices. Completing Proposition 1 would
therefore require developing that theory first.

Further out of reach with current libraries: the bound `L̄_g(t) = O(t²)` needs
uniform control of the pullback Hessian over a compact manifold, and Theorem 2.8
of Boumal, Absil and Cartis needs analysis on Riemannian manifolds. Mathlib has
neither the Stiefel manifold nor the QR retraction.

## Building

```bash
lake exe cache get
lake build
```
