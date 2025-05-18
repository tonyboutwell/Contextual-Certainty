#!/usr/bin/env python3
"""
cct_invariants_demonstration.py

Illustrates the calculation of individual CCT invariants:
- Rapidity (chi_AB)
- Contextual Misalignment (kappa_ABO)
- Geometric Phase (gamma_ABO)

This script shows examples for each, including cases designed
to produce small and large kappa values, with explanations of the results.
"""

import numpy as np
from math import degrees

# ─────────────────────────────────────────────────────────────────────────────
#  Small linear‑algebra toolkit & CCT Invariant Functions (from FR demo)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    """Return v / ‖v‖ (with graceful handling of the zero vector)."""
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

fidelity = lambda a, b: abs(np.vdot(a, b)) ** 2  # |⟨a|b⟩|² for pure states

def chi(a: np.ndarray, b: np.ndarray) -> float:
    """Rapidity chi_AB = -ln(F_AB) for pure states A and B."""
    fid = fidelity(normalize(a), normalize(b))
    if abs(fid - 1) < 1e-15:
        return 0.0
    if fid < 1e-15:
        return np.inf
    return -np.log(fid)

def kappa(a: np.ndarray, b: np.ndarray, o: np.ndarray) -> float:
    """Contextual Misalignment kappa_ABO = sqrt(det G(A,B,O))
       where G_ij = |<psi_i|psi_j>|^2."""
    nA, nB, nO = map(normalize, (a, b, o))
    F_AB, F_AO, F_BO = fidelity(nA, nB), fidelity(nA, nO), fidelity(nB, nO)
    det_G = 1 + 2 * F_AB * F_AO * F_BO - (F_AB**2 + F_AO**2 + F_BO**2)
    return np.sqrt(max(det_G, 0.0))

def gamma(a: np.ndarray, b: np.ndarray, o: np.ndarray) -> float:
    """Bargmann phase gamma_ABO = arg(<A|B><B|O><O|A>).
       Returns 0 if any inner product in the product is near zero."""
    nA, nB, nO = map(normalize, (a, b, o))
    vdot_AB = np.vdot(nA, nB)
    vdot_BO = np.vdot(nB, nO)
    vdot_OA = np.vdot(nO, nA)
    if abs(vdot_AB) < 1e-9 or abs(vdot_BO) < 1e-9 or abs(vdot_OA) < 1e-9:
        return 0.0
    product = vdot_AB * vdot_BO * vdot_OA
    return np.angle(product)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper to print kets and explanations
# ─────────────────────────────────────────────────────────────────────────────
def ket_to_str(v: np.ndarray, precision=3):
    norm_v = normalize(v)
    
    def format_complex(x_complex): # Renamed x to x_complex for clarity
        if abs(x_complex.imag) < 1e-9:
            # Format as real
            return f"{x_complex.real:.{precision}f}"
        else:
            # Format as complex
            real_part_str = f"{x_complex.real:.{precision}f}"
            # Manually add sign for the imaginary part
            if x_complex.imag >= 0:
                imag_part_str = f"+{x_complex.imag:.{precision}f}"
            else:
                imag_part_str = f"{x_complex.imag:.{precision}f}" # Negative sign is already there
            return f"({real_part_str}{imag_part_str}j)"

    # For 1D arrays (kets), we can directly format them to avoid deep NumPy recursion issues
    # if the array is simple enough.
    if norm_v.ndim == 1:
        elements = [format_complex(x) for x in norm_v]
        return "[" + " ".join(elements) + "]"
    else: # Fallback to numpy's formatter for higher-dimensional arrays if needed
        return np.array2string(norm_v, formatter={'complex_kind': format_complex, 
                                                  'float_kind': lambda x_float: f"{x_float:.{precision}f}"})
def print_explanation(text):
    print(f"  Explanation: {text}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  Basis kets for demonstrations
# ─────────────────────────────────────────────────────────────────────────────
q0 = np.array([1, 0], dtype=complex)
q1 = np.array([0, 1], dtype=complex)
plus = normalize(q0 + q1)
minus = normalize(q0 - q1)
plus_i = normalize(q0 + 1j * q1)
minus_i = normalize(q0 - 1j * q1)

q0_perturbed = normalize(q0 + 0.1 * q1)
q0_perturbed_more = normalize(q0_perturbed + 0.1 * q1)


print("="*70)
print(" CCT INVARIANT DEMONSTRATIONS (with Explanations)")
print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
#  Section 1: Rapidity (chi_AB)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Section 1: Rapidity chi_AB = -ln(|<A|B>|^2) ---\n")
print_explanation("Rapidity (chi) measures the 'information distance' or 'distinguishability' between two quantum states A and B. It's derived from their fidelity F_AB = |<A|B>|^2.")

print("Case 1.1: Identical states (A=B)")
A1, B1 = q0, q0
chi_1_1 = chi(A1, B1)
print(f"  A = {ket_to_str(A1)}, B = {ket_to_str(B1)}")
print(f"  Fidelity F_AB = {fidelity(A1,B1):.3f}")
print(f"  chi_AB = {chi_1_1:.3f}")
print_explanation("For identical states, fidelity is 1, so chi = -ln(1) = 0. There's no 'distance' between them.")

print("Case 1.2: Orthogonal states (A perpendicular to B)")
A2, B2 = q0, q1
chi_1_2 = chi(A2, B2)
print(f"  A = {ket_to_str(A2)}, B = {ket_to_str(B2)}")
print(f"  Fidelity F_AB = {fidelity(A2,B2):.3f}")
print(f"  chi_AB = {chi_1_2}")
print_explanation("For orthogonal states, fidelity is 0, so chi = -ln(0) = infinity. They are perfectly distinguishable.")

print("Case 1.3: Non-orthogonal, non-identical states")
A3, B3 = q0, plus
chi_1_3 = chi(A3, B3)
print(f"  A = {ket_to_str(A3)}, B = {ket_to_str(B3)}")
print(f"  Fidelity F_AB = {fidelity(A3,B3):.3f}")
print(f"  chi_AB = {chi_1_3:.3f}")
print_explanation(f"For states with partial overlap (here F_AB=0.5), chi is finite (ln(2) ≈ 0.693). This quantifies their intermediate distinguishability.")

# ─────────────────────────────────────────────────────────────────────────────
#  Section 2: kappa_ABO and gamma_ABO
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Section 2: kappa_ABO and gamma_ABO ---\n")
print_explanation("Contextual Misalignment (kappa_ABO) measures the 'planarity defect' of the triad of states (A, B, O). It's derived from the Gram determinant of their pairwise fidelities. It's zero if the states are 'coplanar' in a generalized sense (e.g., two are identical, or one lies on the geodesic between the other two on the Bloch sphere, if O is one of them).\n  Geometric Phase (gamma_ABO) is the Bargmann invariant, capturing a phase accumulated when traversing the 'triangle' A-B-O-A. It reflects holonomy.")

print("Case 2.1: States A, B, O are 'close' (nearly collinear on Bloch sphere)")
A_k1, B_k1, O_k1 = q0, q0_perturbed, q0_perturbed_more
kappa_2_1 = kappa(A_k1, B_k1, O_k1)
gamma_2_1 = gamma(A_k1, B_k1, O_k1)
print(f"  A = {ket_to_str(A_k1)}")
print(f"  B = {ket_to_str(B_k1)}")
print(f"  O = {ket_to_str(O_k1)}")
print(f"  kappa_ABO = {kappa_2_1:.5f}")
print(f"  gamma_ABO = {gamma_2_1:.3f} rad = {degrees(gamma_2_1):.2f} deg")
print_explanation("When A, B, and O are very close (small perturbations), kappa is very small, indicating minimal misalignment or 'volume' of the triad. Gamma is also near zero as the 'area' of the triangle is small and states are real-valued.")

print("Case 2.2: A, B, O are somewhat 'collinear' on Bloch sphere but distinct")
A_k2, B_k2, O_k2 = q0, plus, q1
kappa_2_2 = kappa(A_k2, B_k2, O_k2)
gamma_2_2 = gamma(A_k2, B_k2, O_k2)
print(f"  A = {ket_to_str(A_k2)}")
print(f"  B = {ket_to_str(B_k2)}")
print(f"  O = {ket_to_str(O_k2)}")
print(f"  F_AB={fidelity(A_k2,B_k2):.3f}, F_AO={fidelity(A_k2,O_k2):.3f}, F_BO={fidelity(B_k2,O_k2):.3f}")
print(f"  kappa_ABO = {kappa_2_2:.5f}")
print(f"  gamma_ABO = {gamma_2_2:.3f} rad = {degrees(gamma_2_2):.2f} deg")
print_explanation("Here A=|0>, B=|+>, O=|1>. Since A and O are orthogonal (F_AO=0), the triangle has one 'zero-length' side in terms of fidelity between A and O. The kappa value is sqrt(0.5) approx 0.707. Gamma is zero because the term <O|A> in its definition is zero.")

print("Case 2.3: A, B, O are widely spread and 'non-planar' (e.g., axes of Bloch sphere)")
A_k3, B_k3, O_k3 = q0, plus, plus_i
kappa_2_3 = kappa(A_k3, B_k3, O_k3)
gamma_2_3 = gamma(A_k3, B_k3, O_k3)
print(f"  A = {ket_to_str(A_k3)}")
print(f"  B = {ket_to_str(B_k3)}")
print(f"  O = {ket_to_str(O_k3)}")
print(f"  F_AB={fidelity(A_k3,B_k3):.3f}, F_AO={fidelity(A_k3,O_k3):.3f}, F_BO={fidelity(B_k3,O_k3):.3f}")
print(f"  kappa_ABO = {kappa_2_3:.5f}")
print(f"  gamma_ABO = {gamma_2_3:.3f} rad = {degrees(gamma_2_3):.2f} deg")
print_explanation("A=|0>, B=|+>, O=|+i> are like axes of the Bloch sphere. All pairwise fidelities are 0.5. This gives kappa = sqrt(0.5) approx 0.707. The non-zero gamma (45 deg) indicates a non-trivial geometric phase for this triad.")

print("Case 2.4: Degenerate case where two states are identical (e.g., B=O)")
A_k4, B_k4, O_k4 = q0, plus, plus
kappa_2_4 = kappa(A_k4, B_k4, O_k4)
gamma_2_4 = gamma(A_k4, B_k4, O_k4)
print(f"  A = {ket_to_str(A_k4)}")
print(f"  B = {ket_to_str(B_k4)}")
print(f"  O = {ket_to_str(O_k4)} (same as B)")
print(f"  kappa_ABO = {kappa_2_4:.5f}")
print(f"  gamma_ABO = {gamma_2_4:.3f} rad = {degrees(gamma_2_4):.2f} deg")
print_explanation("If two states in the triad are identical (here B=O), the 'triangle' degenerates, and kappa becomes 0. This signifies perfect 'planarity' or alignment in this context. Gamma is also zero.")

print("Case 2.5: Another distinct case to observe gamma")
A_k5, B_k5, O_k5 = plus, q1, plus_i
kappa_2_5 = kappa(A_k5, B_k5, O_k5)
gamma_2_5 = gamma(A_k5, B_k5, O_k5)
print(f"  A = {ket_to_str(A_k5)}")
print(f"  B = {ket_to_str(B_k5)}")
print(f"  O = {ket_to_str(O_k5)}")
print(f"  F_AB={fidelity(A_k5,B_k5):.3f}, F_AO={fidelity(A_k5,O_k5):.3f}, F_BO={fidelity(B_k5,O_k5):.3f}")
print(f"  kappa_ABO = {kappa_2_5:.5f}")
print(f"  gamma_ABO = {gamma_2_5:.3f} rad = {degrees(gamma_2_5):.2f} deg")
print_explanation("For A=|+>, B=|1>, O=|+i>, we again find kappa = sqrt(0.5) and gamma = 45 deg. This demonstrates that different triads can have the same CCT invariant values if their geometric relationships are similar.")
