# Exact CCT SU(2) Certainty Transformation Law

This folder provides the validation for the exact Contextual Certainty Transformation (CCT) law for single-qubit (SU(2)) systems.

## Key Files:

*   `cct_su2_transformation_demo.py`: The primary Python script that rigorously validates the SU(2) CCT law:
    `T = 0.5 * [1 + (2*F_AB - 1)*(2*P_A - 1) + 4*sqrt(max(0, F_AB*(1-F_AB)*P_A*(1-P_A))) * cos(gamma_relative_dihedral)]`
    The script confirms that this formula is exact (R² ≈ 1.0) when:
    *   States |A⟩ and |B⟩ are pure qubit states.
    *   `P_A` is the initial certainty `|<O_eig|A>|^2`.
    *   `F_AB` is the fidelity `|<A|B>|^2`.
    *   `gamma_relative_dihedral` is the **oriented (signed) dihedral angle** calculated from the Bloch vectors of A, B, and O_eig.
*   `[cct_su2_transformation_demo_RESULTS.txt]`

## Purpose:
This exact SU(2) law is a foundational component of the CCT framework. It serves as the baseline for the highly accurate SU(N) `KappaAdaptiveCCTPredictor`, which applies this law to optimally projected effective 2D states.

The detailed derivation and discussion of this law can be found in Section 3 of the main project `README.md`.
