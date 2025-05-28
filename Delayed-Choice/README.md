# CCT Resolution of the Delayed Choice Quantum Eraser

This folder contains the demonstration of how the Contextual Certainty Transformation (CCT) framework provides a quantitative, non-retrocausal explanation for Wheeler's Delayed Choice experiment and the Quantum Eraser.

## Key Files:

*   `delayed_choice_cct_demo.py`: The primary Python script. It includes the full `KappaAdaptiveCCTPredictor` class and applies it to various Delayed Choice and Quantum Eraser scenarios. It calculates CCT invariants, predicts measurement probabilities with high accuracy (matching quantum mechanics), and demonstrates how context shifts (due to observer choices) explain the phenomena.
*   `delayed_choice_cct_demo RESULTS.txt`: Text output from a sample run of the demo script, showing detailed analysis for each scenario.
*   `delayed_choice_cct_analysis.png`: Visualization of the results, including CCT invariant values, predictor accuracy, and context-dependent effects.

## Core Insight:
CCT resolves the "paradoxical" nature of delayed choice by showing that the observer's (possibly delayed) choice of measurement setup for an entangled idler particle simply redefines the overall geometric context (`|B_scenario⟩`) for predicting outcomes on the signal particle. The CCT framework quantifies this context shift (`A → B_scenario`) and accurately predicts the signal particle's behavior (wave or particle) without invoking retrocausality. `kappa_ABO` and other CCT invariants track this observer-frame evolution.

Refer to the main project `README.md` for the full CCT theory.
