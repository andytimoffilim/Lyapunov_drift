# Dynamic trajectory stability as a generalization criterion

This repository contains supplementary materials for the theoretical paper:

**"Dynamic trajectory stability as a generalization criterion: Lyapunov's approach to the analysis of stochastic learning"**

The paper develops a framework that connects the stability of the distribution of learning trajectories (under single‑data replacement) to the generalisation gap. The analysis combines Lyapunov drift, martingale concentration inequalities, Wasserstein geometry, and classical algorithmic stability theory.

## Repository structure
Lyapunov_drift/
├── README.md
├── LICENSE
├── requirements.txt
├── paper/
│ └── Lyapunov_drift.pdf # full manuscript in PDF format
├── code/
│ ├── run_synthetic_experiment.py # main experiment script (generates data and figures)
│ └── plot_results.py # optional script to re-plot figures from saved CSV
└── data/
└── synthetic_results.csv # raw numerical results used in Table 1 of the paper

text

- `paper/Lyapunov_drift.pdf` – the full manuscript in PDF format.
- `code/run_synthetic_experiment.py` – the main Python script that runs the full synthetic experiment, computes all indices, and generates the figures.
- `code/plot_results.py` – a helper script that recreates the figures from the saved CSV (useful for quick plotting without re-running the experiment).
- `data/synthetic_results.csv` – the raw numerical results used in Table 1 of the paper.

## Reproducing the results

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
Generate the data and compute the indices:

bash
cd code
python run_synthetic_experiment.py
This script:

generates the synthetic linear regression dataset (d = 20, σ = 0.05)

runs SGD trajectories for sample sizes n ∈ {20, 30, …, 1500}

computes the indices S_T, B_T, C_ε and the generalisation gap

saves the results to synthetic_results.csv

produces the four figures presented in the paper.

(Optional) Re‑plot the figures from the saved CSV:

bash
python plot_results.py
This script loads synthetic_results.csv and regenerates the same figures without re‑running the experiment.

Dependencies
Python ≥ 3.7

NumPy

Matplotlib

SciPy

scikit-learn

pandas

License
The code in this repository is released under the MIT License.
The text of the paper is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).
See the LICENSE file for details.

Citation
If you use this code or the experimental data, please cite the paper:


@article{timofeev2026trajectory,
  title={Dynamic trajectory stability as a generalization criterion: Lyapunov's approach to the analysis of stochastic learning},
  author={Timofeev, Andrey V. and Anufriev, Alexandr S.},
  journal={SIAM Journal on Mathematics of Data Science},
  year={2026},
  note={under review}
}

