import Mathlib.Data.Matrix.Mul

open Matrix

variable {R : Type*} [CommRing R] {p q : ℕ}

/-- Moving an orthonormal projection across a dot product. -/
private theorem dot_compress
    (W : Matrix (Fin q) (Fin p) R) (u : Fin q → R) (z : Fin p → R) :
    u ⬝ᵥ (W *ᵥ z) = (Wᵀ *ᵥ u) ⬝ᵥ z := by
  rw [Matrix.dotProduct_mulVec, Matrix.mulVec_transpose]

/-- Orthonormal compressions preserve quadratic-form lower bounds.

If `S` dominates `c • I` in the quadratic-form order, then so does its
compression `W * S * Wᵀ` for every `W` on the Stiefel manifold.  Taking `c` to
be the least eigenvalue of `S`, this is exactly the Rayleigh-Ritz step used in
Proposition 1: `λ_min (W Σ Wᵀ) ≥ λ_min Σ`.  No spectral theory is required. -/
theorem compression_quadratic_lower_bound [LE R]
    (W : Matrix (Fin q) (Fin p) R) (S : Matrix (Fin p) (Fin p) R)
    (hW : W * Wᵀ = 1) (c : R)
    (hS : ∀ v : Fin p → R, c * (v ⬝ᵥ v) ≤ v ⬝ᵥ (S *ᵥ v)) (u : Fin q → R) :
    c * (u ⬝ᵥ u) ≤ u ⬝ᵥ ((W * S * Wᵀ) *ᵥ u) := by
  have hnorm : (Wᵀ *ᵥ u) ⬝ᵥ (Wᵀ *ᵥ u) = u ⬝ᵥ u := by
    rw [← dot_compress, Matrix.mulVec_mulVec, hW, Matrix.one_mulVec]
  have hquad : u ⬝ᵥ ((W * S * Wᵀ) *ᵥ u) = (Wᵀ *ᵥ u) ⬝ᵥ (S *ᵥ (Wᵀ *ᵥ u)) := by
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, dot_compress]
  rw [hquad, ← hnorm]
  exact hS _
