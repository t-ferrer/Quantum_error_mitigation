import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt

with open(r'C:/Users/thoma/Documents/Travail/QNT/Quantum_error_mitigation/notebooks/_slide_data.pkl', 'rb') as f:
    D = pickle.load(f)
data_3pt, data_5pt = D['data_3pt'], D['data_5pt']
fits_fixed, fits_free = D['fits_fixed'], D['fits_free']
SF3, SF5 = D['SF3'], D['SF5']
SHOW_SEEDS = D['SHOW_SEEDS']
SEED_COLORS = D['SEED_COLORS']
A_FIXED = D['A_FIXED']

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, facecolor='white')

# ---- LEFT: fixed asymptote ----
ax_l.set_facecolor('#fff7f7')
lam_dense = np.linspace(0, 2.2, 400)
for s in SHOW_SEEDS:
    c = SEED_COLORS[s]
    pts = data_3pt[s]
    ax_l.scatter(SF3, pts, color=c, s=85, zorder=4, edgecolor='black', linewidth=0.6)
    status, popt = fits_fixed[s]
    if status == 'ok':
        b, cc = popt
        y = A_FIXED + b * np.exp(-cc * lam_dense)
        ax_l.plot(lam_dense, y, color=c, lw=2.0, alpha=0.85, label=f'seed {s}')
        f0 = A_FIXED + b
        if abs(f0) < 5:
            ax_l.scatter([0], [f0], color=c, s=180, marker='X',
                          edgecolor='black', linewidth=0.9, zorder=5)
ax_l.axhline(A_FIXED, color='#d62728', ls='--', lw=2.2, alpha=0.85,
              label='forced a* = 0.377')
ax_l.axhline(1.0, color='black', ls=':', lw=1.3, alpha=0.6, label='ideal = 1.0')
ax_l.set_xlim(-0.15, 2.25)
ax_l.set_ylim(-0.05, 2.7)
ax_l.set_xlabel('noise scale factor  $\\lambda$', fontsize=12)
ax_l.set_ylabel('P(|00$\\rangle$)', fontsize=12)
ax_l.set_title('Fixed asymptote   $f(\\lambda) = a^* + b\\cdot e^{-c\\lambda}$   (3 scale factors)',
                fontsize=13.5, color='#a00000', fontweight='bold', pad=8)
ax_l.legend(loc='center right', fontsize=9.5, framealpha=0.92)
ax_l.grid(alpha=0.28)

ax_l.annotate('seed 1 fit  $\\rightarrow$  f(0) = 5x10$^9$',
              xy=(0.05, 2.6), xytext=(0.55, 2.45),
              fontsize=11, color='#d62728', fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.2))

ax_l.text(0.08, 1.78,
          'Exp MSE = 1.61\nbias = -0.32\n1/20 catastrophic',
          fontsize=10.5, fontweight='bold', color='#600',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe0e0',
                    edgecolor='#a00', lw=1.5))

# ---- RIGHT: free asymptote ----
ax_r.set_facecolor('#f5fff5')
lam_dense2 = np.linspace(0, 5.5, 400)
asymptotes_per_seed = {}
for s in SHOW_SEEDS:
    c = SEED_COLORS[s]
    pts = data_5pt[s]
    ax_r.scatter(SF5, pts, color=c, s=85, zorder=4, edgecolor='black', linewidth=0.6)
    status, popt = fits_free[s]
    if status == 'ok':
        a_, b, cc = popt
        y = a_ + b * np.exp(-cc * lam_dense2)
        ax_r.plot(lam_dense2, y, color=c, lw=2.0, alpha=0.85,
                  label=f'seed {s}  (a={a_:.2f})')
        ax_r.scatter([0], [a_+b], color=c, s=180, marker='X',
                      edgecolor='black', linewidth=0.9, zorder=5)
        ax_r.axhline(a_, color=c, ls=':', lw=1.0, alpha=0.45, xmin=0.85)
        asymptotes_per_seed[s] = a_

ax_r.axhline(1.0, color='black', ls=':', lw=1.3, alpha=0.6, label='ideal = 1.0')
ax_r.set_xlim(-0.3, 5.7)
ax_r.set_ylim(-0.05, 1.25)
ax_r.set_xlabel('noise scale factor  $\\lambda$', fontsize=12)
ax_r.set_ylabel('P(|00$\\rangle$)', fontsize=12)
ax_r.set_title('Free asymptote   $f(\\lambda) = a + b\\cdot e^{-c\\lambda}$   (5 scale factors)',
                fontsize=13.5, color='#006400', fontweight='bold', pad=8)
ax_r.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax_r.grid(alpha=0.28)

a_vals = [f'{asymptotes_per_seed[s]:.2f}' for s in SHOW_SEEDS]
ax_r.text(2.0, 1.16,
          'per-seed asymptotes:  ' + ', '.join(a_vals) + '    sigma ~ 0.15',
          fontsize=10, color='#333', style='italic')

ax_r.text(0.15, 0.13,
          'Exp MSE = 0.00094\nbias = +0.001\n0/20 failures',
          fontsize=10.5, fontweight='bold', color='#063',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#d6ffd6',
                    edgecolor='#080', lw=1.5))

plt.tight_layout()
OUT = r'C:/Users/thoma/Documents/Travail/QNT/Quantum_error_mitigation/notebooks/zne_free_asymptote_schema.png'
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
import os
print(f'File size: {os.path.getsize(OUT)/1024:.1f} KB')
