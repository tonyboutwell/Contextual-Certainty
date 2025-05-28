# The KappaAdaptiveCCTPredictor

This folder contains the core Python class definition for the `KappaAdaptiveCCTPredictor`, the advanced model used throughout this CCT project for predicting SU(N) certainty transformations.

## Key Files:

*   `kappa_adaptive_cct_predictor.py`: Contains the full Python class `KappaAdaptiveCCTPredictor`. This class implements:
    *   The SU(N) to effective SU(2) projection via Singular Value Decomposition (SVD).
    *   Calculation of 2D parameters (`P_A_2d`, `F_AB_2d`, dihedral `gamma_relative_2d`).
    *   Application of the exact CCT SU(2) law to these projected parameters.
    *   A projection-norm weighting factor (`combined_weight`).
    *   The `kappa_ABO` threshold (≈0.85) based strategy:
        *   For low `kappa_ABO` (N-dimensional): Uses the SVD-projected weighted SU(2) prediction (`T_svd_weighted`).
        *   For high `kappa_ABO`: Applies a learned additive correction (trained via Ridge regression on various N-dimensional geometric features) to `T_svd_weighted`.
    *   Methods for training the correction model (`train_correction_model`) and making predictions (`predict`).
*   `kappa_adaptive_cct_predictor results.txt`: Output from validation runs of this predictor, demonstrating its high accuracy (R² ≈ 0.99) across SU(N) dimensions (N=2 to 10).

## Note:
The demonstration scripts for specific paradoxes (FR, Delayed Choice, Hardy) in other folders embed a copy of this class for self-contained execution, including on-the-fly training of the correction model.

The detailed algorithm and performance are discussed in Section 4 and Appendix C.2 of the main project `README.md`.
