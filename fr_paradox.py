#!/usr/bin/env python3
"""
fr_paradox_cct_demo.py  –  Minimal–yet‑complete live demonstration that
Contextual Certainty Transformations (CCT) resolve the Frauchiger–Renner
(FR) paradox inside standard quantum mechanics.

──────────────────────────────────────────────────────────────────────────────
👩‍🔬  What reviewers get from running this file
──────────────────────────────────────────────────────────────────────────────
1. **A straight‑line reproduction** of the essential two‑qubit fragment of the
   FR thought‑experiment (one Friend + one Wigner, no external lab).  The
   states are built algebraically, so there is *zero* dependence on external
   quantum‑sim packages—NumPy only.

2. **Automatic evaluation of the CCT invariants** (χ, κ, γ) on each logical
   leg of the reasoning loop plus Möbius composition around the loop.  A
   non‑zero loop invariant ⇒ the classical "consistency" step is
   geometrically forbidden, i.e.
   the paradox disappears without leaving the quantum formalism.

3. **A phase‑tweaked variant** in which we add an S‑gate (π/2 Z‑rotation) to
   the coin before the Friend measures.  This injects non‑trivial Bargmann
   phases and shows that CCT still catches the resulting holonomy: χ_loop
   and |z_loop| both grow, and the loop acquires a finite complex argument.

4. The entire script is ~120 lines including extensive comments—short enough
   to read in one sitting, but complete enough that reviewers can copy‑paste
   functions into their own notebooks or circuits.

──────────────────────────────────────────────────────────────────────────────
📚  Quick reference:  the CCT invariants
──────────────────────────────────────────────────────────────────────────────
  χ(A,B)   = –ln F_AB         (rapidity / metric distance)
  κ(A,B,O) = √det G           (planarity‑defect where G_ij = |⟨ψ_i|ψ_j⟩|²)
  γ(A,B,O) = arg⟨A|B⟩⟨B|O⟩⟨O|A⟩   (Bargmann phase / holonomy)
  z        = κ·e^{iγ}         (compactifies κ & γ into a single complex)
  z_loop   = z₁ ⊕ z₂ ⊕ z₃     (Möbius addition; Δ_z terms omitted here)

For the canonical FR parameters the analytic benchmark is
    χ_loop ≈ 2.831      |z_loop| ≈ 0.400, arg 0°
You will see those numbers in the baseline run.
"""

import numpy as np
from math import degrees

# ─────────────────────────────────────────────────────────────────────────────
#  Small linear‑algebra toolkit (pure NumPy)                                   
# ─────────────────────────────────────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    """Return v / ‖v‖ (with graceful handling of the zero vector)."""
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

fidelity = lambda a, b: abs(np.vdot(a, b)) ** 2  # |⟨a|b⟩|² for pure states

chi = lambda a, b: (
    0
    if abs(fidelity(a, b) - 1) < 1e-15              # identical kets → χ=0
    else np.inf if fidelity(a, b) < 1e-15           # orthogonal kets → χ=∞
    else -np.log(fidelity(a, b))                    # general case
)

def kappa(a: np.ndarray, b: np.ndarray, o: np.ndarray) -> float:
    """Quadratic κ from the 3×3 fidelity Gram determinant."""
    nA, nB, nO = map(normalize, (a, b, o))
    F_AB, F_AO, F_BO = fidelity(nA, nB), fidelity(nA, nO), fidelity(nB, nO)
    det = 1 + 2 * F_AB * F_AO * F_BO - (F_AB ** 2 + F_AO ** 2 + F_BO ** 2)
    return np.sqrt(max(det, 0.0))  # numeric guard: det≥0 analytically


def gamma(a: np.ndarray, b: np.ndarray, o: np.ndarray) -> float:
    """Bargmann phase γ ∈ (–π,π]. Returns 0 if triangle has a zero edge."""
    nA, nB, nO = map(normalize, (a, b, o))
    prod = np.vdot(nA, nB) * np.vdot(nB, nO) * np.vdot(nO, nA)
    return 0.0 if abs(prod) < 1e-9 else np.angle(prod)

z_val = lambda a, b, o: kappa(a, b, o) * np.exp(1j * gamma(a, b, o))

# Möbius addition for two and three terms (Δ_z ignored in this demo)
mobius = lambda z1, z2: z1 + z2 - z1 * z2
mobius3 = lambda z1, z2, z3: z1 + z2 + z3 - z1 * z2 - z1 * z3 - z2 * z3 + z1 * z2 * z3

# ─────────────────────────────────────────────────────────────────────────────
#  Basis kets and fixed observable                                            
# ─────────────────────────────────────────────────────────────────────────────
q0, q1 = np.array([1, 0]), np.array([0, 1])
plus = normalize(q0 + q1)

ket00 = np.kron(q0, q0)  # |0⟩_S |0⟩_F
ket10 = np.kron(q1, q0)  # |1⟩_S |0⟩_F
ket11 = np.kron(q1, q1)  # |1⟩_S |1⟩_F

# CCT reference observable  O = |+⟩_S |0⟩_F
O_ket = normalize(np.kron(plus, q0))

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build ψ₁ and ψ₂ from an arbitrary pre‑Friend ψ₀                    
# ─────────────────────────────────────────────────────────────────────────────

def build_psi1_psi2(psi0: np.ndarray):
    """Implement the Friend + Wigner steps of the FR protocol on ψ₀."""
    # Decompose ψ₀ = α|00⟩ + β|10⟩  (Friend's memory still |0⟩)
    α, β = np.vdot(ket00, psi0), np.vdot(ket10, psi0)

    # Friend measures in the X basis. Coefficients for |+⟩ and |−⟩ outcomes:
    a_plus, a_minus = (α + β) / np.sqrt(2), (α - β) / np.sqrt(2)

    # |ψ₁⟩ =  a₊ |+⟩_S |0⟩_F  +  a₋ |1⟩_S |1⟩_F  (tails branch resets S→|1⟩)
    psi1 = normalize(a_plus * np.kron(plus, q0) + a_minus * ket11)

    # Wigner measures "fail" = (|00⟩ - |11⟩)/√2 and post‑selects that branch
    psi2 = normalize(ket00 - ket11)
    return psi1, psi2

# ─────────────────────────────────────────────────────────────────────────────
#  (A) Baseline run – no hidden phase                                         
# ─────────────────────────────────────────────────────────────────────────────
psi0 = normalize(ket00 + np.sqrt(2) * ket10)  # (|00⟩+√2|10⟩)/√3
psi1, psi2 = build_psi1_psi2(psi0)

legs = [
    ("Friend unitary", psi0, psi1),
    ("Wigner post-sel", psi1, psi2),
    ("Consistency jump", psi2, psi0),
]

# Pretty‑print routine --------------------------------------------------------

def loop_report(legs, O, label):
    print(f"\n=== {label} ===")
    print("Leg                 χ       κ       γ (deg)")
    print("────────────────────────────────────────────")
    z_vals, chi_loop = [], 0.0
    for name, A, B in legs:
        χ, κ, γ = chi(A, B), kappa(A, B, O), gamma(A, B, O)
        print(f"{name:<17} {χ:6.3f}  {κ:6.3f}  {degrees(γ):9.2f}")
        z_vals.append(κ * np.exp(1j * γ))
        chi_loop += χ
    z_loop = mobius3(*z_vals)
    print("────────────────────────────────────────────")
    print(f"χ_loop  = {chi_loop:.3f}")
    print(f"|z_loop| = {abs(z_loop):.3f}  arg={degrees(np.angle(z_loop)):.2f}°")

loop_report(legs, O_ket, "Baseline (all overlaps real)")

# ─────────────────────────────────────────────────────────────────────────────
#  (B) Variant – insert an S gate on qubit S before Friend’s measurement      
# ─────────────────────────────────────────────────────────────────────────────
S_gate = np.array([[1, 0], [0, 1j]])           # adds π/2 phase to |1⟩
psi0_phase = np.kron(S_gate, np.eye(2)) @ psi0
psi1_phase, psi2_phase = build_psi1_psi2(psi0_phase)

legs_phase = [
    ("Friend unitary", psi0_phase, psi1_phase),
    ("Wigner post-sel", psi1_phase, psi2_phase),
    ("Consistency jump", psi2_phase, psi0_phase),
]

loop_report(legs_phase, O_ket, "After S-gate on S (non-zero γ)")
