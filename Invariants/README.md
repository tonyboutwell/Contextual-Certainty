# CCT Geometric Invariants Demonstration

This folder contains a script illustrating the calculation and basic behavior of the core Contextual Certainty Transformation (CCT) geometric invariants.

## Key Files:

*   `cct_invariants_demonstration.py`: A Python script that defines and calculates:
    *   **Rapidity (`chi_AB`)**: Based on fidelity, measures "distance" between states.
    *   **Contextual Misalignment (`kappa_ABO`)**: From a Gram determinant of fidelities, measures "planarity defect" or deviation from 2D geometry for a triad of states {A,B,O}.
    *   **Geometric Phase (`gamma_ABO_bargmann`)**: The Bargmann invariant, measures holonomy/curvature.
    The script demonstrates their values for various simple qubit state configurations (e.g., identical, orthogonal, collinear, non-planar) and includes explanations.
*   `cct_invariants_demonstration_results.txt`: Sample output from the script.

## Purpose:
This script serves as a pedagogical introduction to the fundamental geometric quantities used throughout the CCT framework. Their definitions and scaling properties are detailed in Section 2.2 and Appendices A & B of the main project `README.md`.
