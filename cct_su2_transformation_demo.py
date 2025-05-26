#!/usr/bin/env python3
"""
cct_su2_transformation_demo.py

Demonstrates the CCT exact single-qubit (SU(2)) certainty transformation law:
T = 0.5 * [1 + (2F-1)*(2P_A-1) + 4*sqrt(max(0, F(1-F)P_A(1-P_A))) * cos(gamma_relative_dihedral)]

Compares the CCT prediction with direct quantum mechanical calculation.
Uses the ORIENTED DIHEDRAL ANGLE for gamma_relative.
"""

import numpy as np
from math import degrees, acos, sqrt, pi # Added pi for clarity in examples

# ------------------------------------------------------------------------------
# Basic Quantum Functions
# ------------------------------------------------------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    norm_val = np.linalg.norm(v)
    if norm_val < 1e-12: return v
    return v / norm_val

fidelity_states = lambda a, b: abs(np.vdot(normalize(a), normalize(b))) ** 2 # Ensure normalized inputs

def ket_to_str(v: np.ndarray, precision=3):
    norm_v = normalize(v.copy())
    def format_complex(x_complex):
        if abs(x_complex.imag) < 1e-9:
            return f"{x_complex.real:.{precision}f}"
        else:
            real_part_str = f"{x_complex.real:.{precision}f}"
            if x_complex.imag >= 0:
                imag_part_str = f"+{x_complex.imag:.{precision}f}"
            else:
                imag_part_str = f"{x_complex.imag:.{precision}f}"
            return f"({real_part_str}{imag_part_str}j)"
    if norm_v.ndim == 1:
        elements = [format_complex(x) for x in norm_v]
        return "[" + " ".join(elements) + "]"
    else: # Fallback for higher-dimensional arrays (though not used in this SU(2) script)
        return np.array2string(norm_v, precision=precision, suppress_small=True,
                               formatter={'complex_kind': format_complex})


# Basis kets
q0 = np.array([1, 0], dtype=complex)
q1 = np.array([0, 1], dtype=complex)

# ------------------------------------------------------------------------------
# CCT SU(2) Transformation Law (Formula from paper)
# ------------------------------------------------------------------------------
def cct_su2_transform(P_A: float, F_AB: float, gamma_relative_dihedral: float) -> float:
    F_clamped = max(0.0, min(1.0, F_AB))
    P_A_clamped = max(0.0, min(1.0, P_A))
    
    val_2F_minus_1 = 2 * F_clamped - 1.0
    val_2P_A_minus_1 = 2 * P_A_clamped - 1.0
    
    sqrt_arg_val = F_clamped * (1.0 - F_clamped) * P_A_clamped * (1.0 - P_A_clamped)
    
    if sqrt_arg_val < 0:
      if abs(sqrt_arg_val) < 1e-12: 
        sqrt_arg_val = 0.0
      else: 
        raise ValueError(f"sqrt_arg is negative: {sqrt_arg_val}. F_AB={F_AB}, P_A={P_A}")

    sqrt_term_multiplier = 4.0 * sqrt(sqrt_arg_val)

    T = 0.5 * (1.0 + val_2F_minus_1 * val_2P_A_minus_1 + sqrt_term_multiplier * np.cos(gamma_relative_dihedral))
    return np.clip(T, 0.0, 1.0) # Ensure output is a valid probability

# ------------------------------------------------------------------------------
# Bloch Vector and Dihedral Angle Calculation (Corrected gamma_relative)
# ------------------------------------------------------------------------------
def state_to_bloch_vector(state_ket: np.ndarray) -> np.ndarray:
    s = normalize(state_ket.copy()) 
    sigma_x = np.array([[0,1],[1,0]], dtype=complex)
    sigma_y = np.array([[0,-1j],[1j,0]], dtype=complex)
    sigma_z = np.array([[1,0],[0,-1]], dtype=complex)
    rho = np.outer(s, s.conj())
    rx = np.real(np.trace(rho @ sigma_x))
    ry = np.real(np.trace(rho @ sigma_y))
    rz = np.real(np.trace(rho @ sigma_z))
    vec = np.array([rx, ry, rz])
    return normalize(vec) # Pure state Bloch vector has norm 1

def projector_to_bloch_vector(O_proj: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(O_proj)
    O_eig_state = None
    if abs(eigenvalues[1] - 1.0) < 1e-6: # eigh sorts, eigenvalues[1] should be ~1
        O_eig_state = eigenvectors[:, 1]
    elif abs(eigenvalues[0] - 1.0) < 1e-6: # Fallback if sorting is unexpected
         O_eig_state = eigenvectors[:, 0]
    else:
        raise ValueError(f"Projector does not have an eigenvalue ~1. Eigenvalues: {eigenvalues}")
    return state_to_bloch_vector(O_eig_state)

def calculate_gamma_relative_dihedral(A_ket: np.ndarray, B_ket: np.ndarray, O_proj: np.ndarray) -> float:
    r_A = state_to_bloch_vector(A_ket)
    r_B = state_to_bloch_vector(B_ket)
    r_O = projector_to_bloch_vector(O_proj)

    norm_cross_AO = np.linalg.norm(np.cross(r_A, r_O))
    norm_cross_AB = np.linalg.norm(np.cross(r_A, r_B))

    if norm_cross_AO < 1e-9 or norm_cross_AB < 1e-9: # Planes ill-defined or A aligned
        return 0.0 

    n1 = np.cross(r_A, r_O); n1_unit = n1 / np.linalg.norm(n1)
    n2 = np.cross(r_A, r_B); n2_unit = n2 / np.linalg.norm(n2)
    
    cos_gamma = np.clip(np.dot(n1_unit, n2_unit), -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    
    # Determine sign based on orientation r_A.(n1 x n2)
    # r_A is already unit norm from state_to_bloch_vector
    sign_determinant = np.dot(np.cross(n1_unit, n2_unit), r_A) 
    
    if abs(gamma) > 1e-9 and abs(gamma - np.pi) > 1e-9: # Only apply sign if not 0 or pi
        if sign_determinant < 0:
            gamma = -gamma
    return gamma

# ------------------------------------------------------------------------------
# Demonstration
# ------------------------------------------------------------------------------
def run_su2_demo(A_state_orig: np.ndarray, B_state_orig: np.ndarray, O_projector: np.ndarray, case_name: str):
    print(f"\n--- {case_name} ---")
    # Normalize inputs for all calculations
    A_state = normalize(A_state_orig.copy())
    B_state = normalize(B_state_orig.copy())

    print(f"Initial State A: {ket_to_str(A_state)}")
    print(f"Transformed State B: {ket_to_str(B_state)}")
    
    # 1. Initial certainty P_A
    # P_A = Tr(rho_A Pi_O) = <A|Pi_O|A> for pure A.
    # If Pi_O = |O_eig><O_eig|, then P_A = |<O_eig|A>|^2
    # This assumes O_projector is indeed a rank-1 projector for |O_eig>.
    # Let's get |O_eig> from O_projector to be sure.
    eigenvalues_O, eigenvectors_O = np.linalg.eigh(O_projector)
    O_eig_state = None
    if abs(eigenvalues_O[1] - 1.0) < 1e-6: O_eig_state = eigenvectors_O[:, 1]
    elif abs(eigenvalues_O[0] - 1.0) < 1e-6: O_eig_state = eigenvectors_O[:, 0]
    else: raise ValueError("O_projector is not a valid rank-1 projector for observable.")
    O_eig_state = normalize(O_eig_state)

    P_A = fidelity_states(A_state, O_eig_state) # Same as |<O_eig|A>|^2
    print(f"Initial Certainty P_A = |<O_eig|A>|^2 = {P_A:.4f}")

    # 2. Fidelity F_AB
    F_AB = fidelity_states(A_state, B_state) # Uses normalized inputs
    print(f"Fidelity F_AB = |<A|B>|^2 = {F_AB:.4f}")

    # 3. Calculate gamma_relative_dihedral independently
    gamma_rel = calculate_gamma_relative_dihedral(A_state, B_state, O_projector)
    print(f"Calculated dihedral gamma_relative = {gamma_rel:.4f} rad = {degrees(gamma_rel):.2f} deg")
    
    # 4. Use CCT SU(2) formula to predict T
    T_CCT = cct_su2_transform(P_A, F_AB, gamma_rel)
    print(f"CCT Predicted Certainty T_CCT = {T_CCT:.4f}")

    # 5. Direct quantum mechanical calculation of T
    T_QM = fidelity_states(B_state, O_eig_state) # |<O_eig|B>|^2
    print(f"Direct QM Certainty T_QM = |<O_eig|B>|^2 = {T_QM:.4f}")

    # 6. Comparison
    diff = abs(T_CCT - T_QM)
    print(f"Difference |T_CCT - T_QM| = {diff:.6f}")
    if diff < 1e-5: # Adjusted tolerance slightly for floating point operations
        print("SUCCESS: CCT prediction matches QM calculation.")
    else:
        print("FAILURE: CCT prediction does NOT match QM calculation.")
    print("-"*(len(case_name) + 8))

if __name__ == "__main__":
    print("="*70)
    print(" CCT SU(2) CERTAINTY TRANSFORMATION LAW DEMONSTRATION (DIHEDRAL ANGLE)")
    print("="*70)

    # Define Observable Projector Pi_O = |0><0|
    Pi_O_0 = np.outer(q0, q0.conj()) 

    # Case 1: Simple phase shift
    A1 = (q0 + q1) # Will be normalized in run_su2_demo
    U1 = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex)
    B1 = U1 @ A1
    run_su2_demo(A1, B1, Pi_O_0, "Case 1: Simple phase shift on |1> component")

    # Case 2: Rotation around Y-axis
    A2 = q0
    theta_y = np.pi/3
    Ry = np.array([[np.cos(theta_y/2), -np.sin(theta_y/2)],
                   [np.sin(theta_y/2),  np.cos(theta_y/2)]], dtype=complex)
    B2 = Ry @ A2
    run_su2_demo(A2, B2, Pi_O_0, f"Case 2: A=|0>, B=Ry({degrees(theta_y):.0f} deg)A")

    # Case 3: General unitary, observable |+><+|
    A3 = (q0 + 0.5j * q1)
    a_u = np.exp(1j*0.3)/sqrt(2) # For |a|^2+|b|^2=1
    b_u = np.exp(-1j*0.7)/sqrt(2)
    U3 = np.array([[a_u, -np.conj(b_u)], [b_u, np.conj(a_u)]], dtype=complex)
    B3 = U3 @ A3
    Pi_O_plus = np.outer(normalize(q0+q1), normalize(q0+q1).conj())
    run_su2_demo(A3, B3, Pi_O_plus, "Case 3: General U, Observable |+><+|")

    # Case 4: Test edge case F_AB=1 (B is A up to a global phase)
    A4 = (q0 + (0.5+0.5j)*q1)
    B4 = np.exp(1j * np.pi/7) * A4 
    run_su2_demo(A4, B4, Pi_O_0, "Case 4: B = global_phase * A (F_AB=1)")
    
    # Case 5: Test edge case F_AB=0 (B orthogonal to A)
    A5 = q0
    B5 = q1
    run_su2_demo(A5, B5, Pi_O_0, "Case 5: B orthogonal to A (F_AB=0)")

    # Case 6: Test P_A = 1
    A6 = q0
    U6_theta = pi/6 # For Ry(30 deg)
    U6 = np.array([[np.cos(U6_theta/2), -np.sin(U6_theta/2)],
                       [np.sin(U6_theta/2),  np.cos(U6_theta/2)]], dtype=complex) 
    B6 = U6 @ A6
    run_su2_demo(A6, B6, Pi_O_0, "Case 6: P_A = 1 (A is eigenstate of O)")

    # Case 7: Test P_A = 0
    A7 = q1
    U7_theta = pi/5 # For Ry(36 deg)
    U7 = np.array([[np.cos(U7_theta/2), -np.sin(U7_theta/2)],
                       [np.sin(U7_theta/2),  np.cos(U7_theta/2)]], dtype=complex) 
    B7 = U7 @ A7
    run_su2_demo(A7, B7, Pi_O_0, "Case 7: P_A = 0 (A is other eigenstate of O)")
    
    # Case 8: More general case from successful run
    A8 = normalize(0.6*q0 + (0.8)*np.exp(1j*np.pi/5)*q1)
    U8_alpha = np.exp(1j*np.pi/3) * np.cos(np.pi/7)
    U8_beta = np.exp(1j*np.pi/4) * np.sin(np.pi/7)
    U8 = np.array([[U8_alpha, -np.conj(U8_beta)],[U8_beta, np.conj(U8_alpha)]], dtype=complex)
    B8 = U8 @ A8
    run_su2_demo(A8, B8, Pi_O_0, "Case 8: General A, general U")