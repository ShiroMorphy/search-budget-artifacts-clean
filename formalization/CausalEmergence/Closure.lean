import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic.NoncommRing

open Matrix

variable {R : Type*} [CommRing R] {p q : ℕ}

/-- Exact decomposition of the projected one-step map.  Projecting the micro
dynamics `x ↦ A x` by an orthonormal `W` does NOT give an autonomous macro
system: an exact residual term `W * A * (1 - Wᵀ * W)` always appears. -/
theorem projected_step_decomposition
    (W : Matrix (Fin q) (Fin p) R) (A : Matrix (Fin p) (Fin p) R) (x : Fin p → R) :
    W *ᵥ (A *ᵥ x)
      = (W * A * Wᵀ) *ᵥ (W *ᵥ x) + (W * A * (1 - Wᵀ * W)) *ᵥ x := by
  rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, ← Matrix.add_mulVec]
  congr 1
  rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc]
  abel

/-- The projected state is an autonomous linear system for every initial
condition **iff** the residual operator vanishes.  This is the exact
"dynamical closure" condition. -/
theorem autonomous_iff_closure
    (W : Matrix (Fin q) (Fin p) R) (A : Matrix (Fin p) (Fin p) R) :
    (∀ x : Fin p → R, W *ᵥ (A *ᵥ x) = (W * A * Wᵀ) *ᵥ (W *ᵥ x))
      ↔ W * A * (1 - Wᵀ * W) = 0 := by
  constructor
  · intro h
    have key : ∀ x : Fin p → R, (W * A * (1 - Wᵀ * W)) *ᵥ x = 0 := by
      intro x
      have hd := projected_step_decomposition W A x
      rw [h x] at hd
      have hz : (W * A * Wᵀ) *ᵥ (W *ᵥ x) + 0
          = (W * A * Wᵀ) *ᵥ (W *ᵥ x) + (W * A * (1 - Wᵀ * W)) *ᵥ x := by
        rw [add_zero]; exact hd
      exact (add_left_cancel hz).symm
    ext i j
    have hij := congrArg (fun v => v i) (key (Pi.single j 1))
    simpa using hij
  · intro h x
    have hd := projected_step_decomposition W A x
    rw [h] at hd
    simpa using hd

/-- Sufficient condition for exact closure: the row space of `W` is invariant
under `A`.  This is the precise sense in which the projected macro model is a
genuine dynamics rather than a Galerkin surrogate. -/
theorem closure_of_left_invariant
    (W : Matrix (Fin q) (Fin p) R) (A : Matrix (Fin p) (Fin p) R)
    (hW : W * Wᵀ = 1) (C : Matrix (Fin q) (Fin q) R) (hC : W * A = C * W) :
    W * A * (1 - Wᵀ * W) = 0 := by
  rw [hC, Matrix.mul_sub, Matrix.mul_one, Matrix.mul_assoc C W (Wᵀ * W),
    ← Matrix.mul_assoc W Wᵀ W, hW, Matrix.one_mul, sub_self]

/-- Compressions of symmetric operators are symmetric. -/
theorem transpose_compression
    (W : Matrix (Fin q) (Fin p) R) (S : Matrix (Fin p) (Fin p) R) :
    (W * S * Wᵀ)ᵀ = W * Sᵀ * Wᵀ := by
  rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.mul_assoc]

/-- Behaviour of the macro pair under a common scalar rescaling `x ↦ c • x`:
the transition operator is unchanged and the innovation covariance picks up
`c ^ 2`.  This is the algebraic content of the scale-invariance claim. -/
theorem compression_smul
    (W : Matrix (Fin q) (Fin p) R) (S : Matrix (Fin p) (Fin p) R) (c : R) :
    W * (c ^ 2 • S) * Wᵀ = c ^ 2 • (W * S * Wᵀ) := by
  rw [Matrix.mul_smul, Matrix.smul_mul]
