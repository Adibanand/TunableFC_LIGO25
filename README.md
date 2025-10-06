# Tunable Filter Cavity for Broadband Squeezing (`tunable-fc-sqz`)

> **Disclaimer:**  
> This public repository contains a limited, cleaned, and de-identified subset of my original research code developed as part of the *Quantum Interferometry @ Berkeley* project.  
> All data, parameters, and scripts included here are for demonstration and educational purposes only, and do **not** include any proprietary or restricted material from the LIGO Scientific Collaboration or affiliated research groups.


Tools to (1) run quantum-noise budgeting with LIGO's GWINC module and optimize **effective transmissivity** $T_\mathrm{eff}\$, then (2) design a **thermally tunable effective two-mirror filter cavity** by solving for $\{R_{1}, R_{2}, L_{1}\}$ that achieve target bandwidth and tunability.

## What this repo does

- **optimizers: `params_objective.py`**  
  Runs a quantum noise budget (QNB) loop to optimize $T_\mathrm{eff}\$ (or related parameters like input coupler ($T_{i}$), detuning, etc.) for a chosen interferometer configuration. Objective functions include minimizing broadband quantum noise across a specified frequency range or maximizing BNS inspiral range.

** See lab notes PDF file for more information regarding optimization of the tunable FC for the tabletop and 300 m configurations
