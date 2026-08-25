markdown
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
│ └── v3_Dynamic_trajectory_stability_as_a_generalization_criterion.docx.pdf
├── code/
│ └── analysis.py
└── data/
└── synthetic_results.csv

text

- `paper/` – the full manuscript in PDF format.
- `code/analysis.py` – Python script that runs the synthetic experiment and generates all figures.
- `data/synthetic_results.csv` – raw numerical results used in Table 1 of the paper.

## Reproducing the results

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
Navigate to the code/ directory and run:

bash
python analysis.py
The script will:

generate the synthetic linear regression dataset (d = 20, σ = 0.05)

run SGD trajectories for sample sizes n ∈ {20, 30, …, 1500}

compute the indices 
ST,BT,Cε and the generalisation gap

produce the three figures presented in the paper.

The script also saves the results in synthetic_results.csv (already provided).

Dependencies
Python ≥ 3.7

NumPy

Matplotlib

SciPy

scikit-learn

License
The code in this repository is released under the MIT License.
The text of the paper is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).
See the LICENSE file for details.