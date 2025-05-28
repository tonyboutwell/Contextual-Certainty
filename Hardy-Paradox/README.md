# CCT Resolution of Hardy's Paradox

This folder contains the demonstration of how the Contextual Certainty Transformation (CCT) framework quantitatively explains and resolves Hardy's Paradox.

## Key Files:

*   `hardy_paradox_cct_demo.py`: The main Python script. It includes the full `KappaAdaptiveCCTPredictor` and applies it to the SU(4) system of entangled particles in Hardy's setup. It predicts conditional and joint probabilities, comparing them to classical expectations and exact quantum mechanics.
*   `hardy_paradox_cct_demo RESULTS.txt`: Sample output detailing the CCT analysis, including predicted probabilities and invariant values for different context shifts within the paradox.
*   `hardy_paradox_cct_analysis.png`: Visualization of the results, comparing classical, quantum, and CCT probabilities, and showing CCT invariant behavior.

## Core Insight:
Hardy's Paradox highlights "impossible" joint measurement outcomes that arise from seemingly logical classical inferences about entangled particles. CCT resolves this by:
1.  Accurately predicting all quantum mechanical probabilities using the `KappaAdaptiveCCTPredictor`.
2.  Showing that each measurement outcome on one particle establishes a new, distinct geometric context for predictions about the other particle.
3.  The CCT invariants (`kappa_ABO`, `chi_AB`, `gamma_ABO_Bargmann`) quantify these context shifts. High `kappa_ABO` values for the conditional scenarios in Hardy's confirm their non-classical, higher-dimensional geometric nature, invalidating classical joint probability assumptions.

Refer to the main project `README.md` for the full CCT theory.
