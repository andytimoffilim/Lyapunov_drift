#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic experiment for the paper:
"Dynamic trajectory stability as a generalization criterion: Lyapunov's approach to the analysis of stochastic learning"

This script reproduces all numerical results presented in the paper:
- Generates synthetic linear regression data (d = 20, true coefficients decaying as 1/j)
- Runs SGD with step size eta = 0.01 for T = 300 steps
- Computes the indices S_T, B_T, C_eps for varying sample sizes n
- Evaluates the generalization gap (test MSE - train MSE)
- Saves results to 'synthetic_results.csv'
- Produces four figures:
    * ST_vs_inv_n_final.png   : S_T vs 1/n with linear fit
    * gap_vs_inv_n_final.png  : generalization gap vs 1/n
    * ST_vs_gap_final.png     : S_T vs gap with correlation
    * B_and_C_fixed.png       : B_T and C_eps vs n

Parameters can be adjusted in the main block.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import time

# ---------------------------
# 1. Data generation
# ---------------------------
def generate_linear_data(n, d, theta_star, sigma=0.1, seed=None):
    """
    Generate n i.i.d. samples from y = theta_star^T x + noise.
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(n, d)
    y = X @ theta_star + sigma * np.random.randn(n)
    return X, y


# ---------------------------
# 2. SGD trajectory (full history)
# ---------------------------
def sgd_trajectory(X, y, theta0, eta, T, batch_size=1, seed=None):
    """
    Run SGD for T steps, storing all parameter vectors.
    """
    if seed is not None:
        np.random.seed(seed)
    n, d = X.shape
    theta = theta0.copy()
    traj = [theta.copy()]
    for t in range(T):
        idx = np.random.choice(n, size=batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        grad = X_batch.T @ (X_batch @ theta - y_batch) / batch_size
        theta = theta - eta * grad
        traj.append(theta.copy())
    return np.array(traj)


# ---------------------------
# 3. Compute indices with fixed R and eps (based on reference n)
# ---------------------------
def compute_R_eps_from_reference(n_ref, d, theta_star, sigma, eta, T, t0, probe_X,
                                 n_runs=30, seed_base=42):
    """
    Estimate R and epsilon from a reference dataset of size n_ref.
    Returns the 90th percentiles of the maximal distances to theta_star
    and of the sup-norm deviations from the mean output trajectory.
    """
    X_ref, y_ref = generate_linear_data(n_ref, d, theta_star, sigma, seed=seed_base)
    theta0 = np.zeros(d)
    trajectories = []
    for r in range(n_runs):
        traj = sgd_trajectory(X_ref, y_ref, theta0, eta, T, batch_size=1,
                              seed=seed_base + 10 + r)
        trajectories.append(traj)

    # R: 90th percentile of max distances to theta_star
    max_dists = []
    for traj in trajectories:
        dists = np.linalg.norm(traj[t0:] - theta_star, axis=1)
        max_dists.append(np.max(dists))
    R = np.percentile(max_dists, 90)

    # epsilon: 90th percentile of sup deviations from mean output
    outputs = []
    for traj in trajectories:
        Y = traj @ probe_X.T
        outputs.append(Y)
    mean_Y = np.mean(outputs, axis=0)
    sup_devs = []
    for Y in outputs:
        dev = np.max(np.linalg.norm(Y - mean_Y, axis=1))
        sup_devs.append(dev)
    eps = np.percentile(sup_devs, 90)

    return R, eps


def compute_indices_fixed_thresholds(trajectories, theta_star, probe_X, t0, R, eps):
    """
    Compute B_T and C_eps using fixed R and eps.
    """
    N_runs = len(trajectories)
    exits = 0
    for traj in trajectories:
        dists = np.linalg.norm(traj[t0:] - theta_star, axis=1)
        if np.any(dists > R):
            exits += 1
    B_T = 1 - exits / N_runs

    outputs = []
    for traj in trajectories:
        Y = traj @ probe_X.T
        outputs.append(Y)
    mean_Y = np.mean(outputs, axis=0)
    sup_devs = []
    for Y in outputs:
        dev = np.max(np.linalg.norm(Y - mean_Y, axis=1))
        sup_devs.append(dev)
    C_eps = np.mean(np.array(sup_devs) > eps)

    return B_T, C_eps


# ---------------------------
# 4. S_T via final parameter difference (coupled)
# ---------------------------
def compute_S_T_final_params(X, y, theta0, eta, T, probe_X, sigma,
                             n_runs=30, n_samples=10, seed_base=0):
    """
    Compute S_T as the average Euclidean distance between final parameters
    of coupled reference and modified trajectories (single data replacement).
    """
    n, d = X.shape
    d = int(d)
    seeds = [seed_base + r for r in range(n_runs)]

    # Reference final parameters
    ref_final = []
    for seed in seeds:
        traj = sgd_trajectory(X, y, theta0, eta, T, batch_size=1, seed=seed)
        ref_final.append(traj[-1])

    dist_sum = 0.0
    for s in range(n_samples):
        i = np.random.randint(n)
        X_i = X.copy()
        y_i = y.copy()
        new_x = np.random.randn(d)
        new_y = new_x @ theta_star + sigma * np.random.randn()
        X_i[i] = new_x
        y_i[i] = new_y

        mod_final = []
        for seed in seeds:
            traj = sgd_trajectory(X_i, y_i, theta0, eta, T, batch_size=1,
                                  seed=seed + 10000 + s * 1000)
            mod_final.append(traj[-1])

        for r in range(n_runs):
            dist = np.linalg.norm(ref_final[r] - mod_final[r])
            dist_sum += dist

    S_T = dist_sum / (n_samples * n_runs)
    return S_T


# ---------------------------
# 5. Main experiment
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()

    # ---------- Parameters ----------
    d = 20
    theta_star = np.array([1.0 / (i + 1) for i in range(d)])  # decaying coefficients
    sigma = 0.05
    eta = 0.01
    T = 300
    t0 = 100
    n_runs = 30
    batch_size = 1
    probe_size = 30

    n_list = [20, 30, 50, 70, 100, 130, 180, 250, 350, 500, 700, 1000, 1500]

    np.random.seed(42)
    probe_X = np.random.randn(probe_size, d)

    # Estimate fixed R and eps from a reference dataset (n_ref=200)
    n_ref = 200
    R_fixed, eps_fixed = compute_R_eps_from_reference(n_ref, d, theta_star, sigma,
                                                      eta, T, t0, probe_X,
                                                      n_runs=n_runs, seed_base=42)
    print(f"Fixed R = {R_fixed:.3f}, eps = {eps_fixed:.3f}")

    results = []

    for idx, n in enumerate(n_list):
        print(f"Processing n={n}...")
        seed_data = 42 + idx * 100

        X_train, y_train = generate_linear_data(n, d, theta_star, sigma,
                                                seed=seed_data)
        X_test, y_test = generate_linear_data(1000, d, theta_star, sigma,
                                              seed=seed_data + 1)

        theta0 = np.zeros(d)

        # Multiple trajectories
        trajectories = []
        for r in range(n_runs):
            traj = sgd_trajectory(X_train, y_train, theta0, eta, T, batch_size,
                                  seed=seed_data + 10 + r)
            trajectories.append(traj)

        # B_T and C_eps with fixed thresholds
        B_T, C_eps = compute_indices_fixed_thresholds(
            trajectories, theta_star, probe_X, t0, R_fixed, eps_fixed
        )

        # S_T via final parameter difference
        S_T = compute_S_T_final_params(
            X_train, y_train, theta0, eta, T, probe_X, sigma,
            n_runs=n_runs, n_samples=10, seed_base=seed_data + 200
        )

        # Generalization gap
        test_preds = [traj[-1] @ X_test.T for traj in trajectories]
        train_preds = [traj[-1] @ X_train.T for traj in trajectories]
        test_mse = np.mean([mean_squared_error(y_test, pred) for pred in test_preds])
        train_mse = np.mean([mean_squared_error(y_train, pred) for pred in train_preds])
        gap = test_mse - train_mse

        results.append({
            'n': n,
            'B_T': B_T,
            'C_eps': C_eps,
            'S_T': S_T,
            'gap': gap
        })
        print(f"  n={n:4d}, B={B_T:.3f}, C={C_eps:.3f}, S={S_T:.3f}, gap={gap:.5f}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f} seconds.")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('synthetic_results.csv', index=False)
    print("Results saved to synthetic_results.csv")

    # ---------------------------
    # Plotting (same as in the paper)
    # ---------------------------
    n_vals = np.array([r['n'] for r in results])
    S_vals = np.array([r['S_T'] for r in results])
    gap_vals = np.array([r['gap'] for r in results])
    B_vals = np.array([r['B_T'] for r in results])
    C_vals = np.array([r['C_eps'] for r in results])

    # 1. S_T vs 1/n
    plt.figure(figsize=(5, 4))
    inv_n = 1.0 / n_vals
    plt.scatter(inv_n, S_vals, color='green')
    coeffs = np.polyfit(inv_n, S_vals, 1)
    poly = np.poly1d(coeffs)
    inv_line = np.linspace(min(inv_n), max(inv_n), 100)
    plt.plot(inv_line, poly(inv_line), 'r--',
             label=f'Slope={coeffs[0]:.2f}, r={np.corrcoef(inv_n, S_vals)[0,1]:.3f}')
    plt.xlabel('$1/n$')
    plt.ylabel('$S_T$')
    plt.legend()
    plt.grid(True)
    plt.title('S_T vs 1/n (final params)')
    plt.tight_layout()
    plt.savefig('ST_vs_inv_n_final.png', dpi=150)
    plt.show()

    # 2. Gap vs 1/n
    plt.figure(figsize=(5, 4))
    plt.scatter(inv_n, gap_vals, color='purple')
    coeffs_g = np.polyfit(inv_n, gap_vals, 1)
    poly_g = np.poly1d(coeffs_g)
    plt.plot(inv_line, poly_g(inv_line), 'r--',
             label=f'Slope={coeffs_g[0]:.3f}, r={np.corrcoef(inv_n, gap_vals)[0,1]:.3f}')
    plt.xlabel('$1/n$')
    plt.ylabel('Gap')
    plt.legend()
    plt.grid(True)
    plt.title('Gap vs 1/n')
    plt.tight_layout()
    plt.savefig('gap_vs_inv_n_final.png', dpi=150)
    plt.show()

    # 3. S_T vs gap (scatter)
    plt.figure(figsize=(5, 4))
    plt.scatter(S_vals, gap_vals, color='blue')
    coeffs_sg = np.polyfit(S_vals, gap_vals, 1)
    poly_sg = np.poly1d(coeffs_sg)
    S_line = np.linspace(min(S_vals), max(S_vals), 100)
    plt.plot(S_line, poly_sg(S_line), 'r--',
             label=f'corr = {np.corrcoef(S_vals, gap_vals)[0,1]:.3f}')
    plt.xlabel('$S_T$')
    plt.ylabel('Gap')
    plt.legend()
    plt.grid(True)
    plt.title('S_T vs Gap')
    plt.tight_layout()
    plt.savefig('ST_vs_gap_final.png', dpi=150)
    plt.show()

    # 4. B_T and C_eps vs n
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(n_vals, B_vals, 'o-')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('B_T')
    axs[0].set_title('Basin retention (fixed R)')
    axs[0].grid(True)
    axs[1].plot(n_vals, C_vals, 'o-')
    axs[1].set_xlabel('n')
    axs[1].set_ylabel('C_eps')
    axs[1].set_title('Concentration (fixed eps)')
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig('B_and_C_fixed.png', dpi=150)
    plt.show()

    print("All plots saved.")