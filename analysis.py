import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Загружаем данные
df = pd.read_csv('synthetic_results.csv')
n = df['n'].values
S_T = df['S_T'].values
gap = df['gap'].values
B_T = df['B_T'].values
C_eps = df['C_eps'].values

# Корреляции
corr_S_gap, p_S = pearsonr(S_T, gap)
corr_B_gap, p_B = pearsonr(B_T, gap)
corr_C_gap, p_C = pearsonr(C_eps, gap)

print(f"Корреляция S_T vs gap: {corr_S_gap:.3f} (p={p_S:.3f})")
print(f"Корреляция B_T vs gap: {corr_B_gap:.3f} (p={p_B:.3f})")
print(f"Корреляция C_eps vs gap: {corr_C_gap:.3f} (p={p_C:.3f})")

# График 1: S_T vs 1/n
plt.figure(figsize=(5,4))
inv_n = 1.0 / n
plt.scatter(inv_n, S_T, color='green')
coeffs = np.polyfit(inv_n, S_T, 1)
poly = np.poly1d(coeffs)
inv_line = np.linspace(min(inv_n), max(inv_n), 100)
plt.plot(inv_line, poly(inv_line), 'r--', label=f'Slope={coeffs[0]:.3f}, r={np.corrcoef(inv_n, S_T)[0,1]:.3f}')
plt.xlabel('$1/n$')
plt.ylabel('$S_T$')
plt.legend()
plt.grid(True)
plt.title('S_T vs 1/n')
plt.tight_layout()
plt.savefig('ST_vs_inv_n_final.png', dpi=150)
plt.show()

# График 2: S_T vs gap
plt.figure(figsize=(5,4))
plt.scatter(S_T, gap, color='blue')
coeffs2 = np.polyfit(S_T, gap, 1)
poly2 = np.poly1d(coeffs2)
S_line = np.linspace(min(S_T), max(S_T), 100)
plt.plot(S_line, poly2(S_line), 'r--', label=f'corr = {corr_S_gap:.3f}')
plt.xlabel('$S_T$')
plt.ylabel('Gap')
plt.legend()
plt.grid(True)
plt.title('S_T vs Generalization Gap')
plt.tight_layout()
plt.savefig('ST_vs_gap_final.png', dpi=150)
plt.show()

# График 3: B_T и C_eps vs n (отдельно)
fig, axs = plt.subplots(1, 2, figsize=(10,4))
axs[0].plot(n, B_T, 'o-', color='orange')
axs[0].set_xlabel('n')
axs[0].set_ylabel('B_T')
axs[0].set_title('Basin retention')
axs[0].grid(True)
axs[1].plot(n, C_eps, 'o-', color='red')
axs[1].set_xlabel('n')
axs[1].set_ylabel('C_eps')
axs[1].set_title('Concentration')
axs[1].grid(True)
plt.tight_layout()
plt.savefig('B_and_C_vs_n_final.png', dpi=150)
plt.show()

# Дополнительно: gap vs 1/n
plt.figure(figsize=(5,4))
plt.scatter(inv_n, gap, color='purple')
coeffs3 = np.polyfit(inv_n, gap, 1)
poly3 = np.poly1d(coeffs3)
plt.plot(inv_line, poly3(inv_line), 'r--', label=f'corr = {np.corrcoef(inv_n, gap)[0,1]:.3f}')
plt.xlabel('$1/n$')
plt.ylabel('Gap')
plt.legend()
plt.grid(True)
plt.title('Gap vs 1/n')
plt.tight_layout()
plt.savefig('gap_vs_inv_n_final.png', dpi=150)
plt.show()