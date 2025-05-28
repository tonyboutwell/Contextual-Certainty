# CCT Resolution of the Frauchiger-Renner Paradox

This folder demonstrates the application of the Contextual Certainty Transformation (CCT) framework to resolve the Frauchiger-Renner (FR) paradox.

## Key Files:

*   `fr_paradox_full_cct_demo.py`: The primary Python script. It embeds the `KappaAdaptiveCCTPredictor` class and uses it to analyze the certainty transformations at each step of the FR reasoning loop. It also calculates the overall CCT loop invariants (`chi_loop`, `z_loop`).
*   `fr_paradox_full_cct_demo RESULTS.txt`: Text output from a sample run, detailing the CCT analysis for different FR scenarios.
*   `fr_paradox_full_cct_analysis.png`: Visualization of the results, including loop holonomies, predictor accuracy, and CCT invariant distributions.

## Core Insight:
The FR paradox arises from assuming perfect consistency of inferences across different (nested) observational contexts. CCT resolves this by:
1.  Accurately predicting the certainty transformations at each step using the `KappaAdaptiveCCTPredictor`.
2.  Demonstrating that the full FR reasoning loop acquires a non-zero geometric "cost" or "holonomy" (non-zero `chi_loop` and `z_loop = kappa_loop * exp(i*gamma_Bargmann_loop)`), calculated via the CCT composition law. This non-closure of the loop within the CCT geometric framework eliminates the logical contradiction from within standard quantum mechanics.

Refer to the main project `README.md` for the full CCT theory.
