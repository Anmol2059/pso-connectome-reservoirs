"""
Step 01 — Load & Explore Real Connectomes
==========================================
Sources (via netneurotools fetch_famous_gmat):
  fly      → Chiang et al. 2011, Drosophila brain, 49 regions
  mouse    → Rubinov et al. 2015, mouse cortex, 112 regions
  rat      → Bota et al. 2015,   rat cortex, 73 regions
  macaque  → Modha & Singh 2010, macaque brain, 242 regions

Outputs:
  images/01_matrix_heatmaps.png
  images/02_degree_distributions.png
"""
import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── paths ─────────────────────────────────────────────────────────────────────
PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(PROJ_DIR, 'data')
IMAGES_DIR = os.path.join(PROJ_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

SPECIES = ['fly', 'mouse', 'rat', 'macaque']

# Actual dataset references — used in figure labels
REFERENCES = {
    'fly':     'Chiang et al. 2011\n(Drosophila, 49 regions)',
    'mouse':   'Rubinov et al. 2015\n(Mouse cortex, 112 regions)',
    'rat':     'Bota et al. 2015\n(Rat cortex, 73 regions)',
    'macaque': 'Modha & Singh 2010\n(Macaque, 242 regions)',
}

# ── load ──────────────────────────────────────────────────────────────────────
connectomes = {}
for sp in SPECIES:
    w      = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    labels = np.load(os.path.join(DATA_DIR, sp, 'labels.npy'))
    connectomes[sp] = {'w': w, 'labels': labels}

# ── print stats ───────────────────────────────────────────────────────────────
print(f'\n{"Species":<10} {"Nodes":>7} {"Edges":>7} {"Density":>9} '
      f'{"Symmetric":>10} {"w_min":>8} {"w_max":>8} {"Input nodes":>12}')
print('-' * 80)
for sp in SPECIES:
    w, labels = connectomes[sp]['w'], connectomes[sp]['labels']
    n         = w.shape[0]
    edges     = np.count_nonzero(w)
    density   = edges / (n * (n - 1))
    sym       = np.allclose(w, w.T)
    w_vals    = w[w > 0]
    n_in      = np.sum(labels == 1)
    print(f'{sp:<10} {n:>7} {edges:>7} {density:>9.4f} '
          f'{str(sym):>10} {w_vals.min():>8.4f} {w_vals.max():>8.2f} {n_in:>12}')

# ── figure 01: matrix heatmaps ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Biological Connectome Weight Matrices', fontsize=14, fontweight='bold')

for ax, sp in zip(axes, SPECIES):
    w = connectomes[sp]['w']
    # log-scale for fly (weights up to 4476); linear for others
    im = ax.imshow(np.log1p(w) if sp == 'fly' else w,
                   cmap='Blues', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(REFERENCES[sp], fontsize=9, fontweight='bold')
    ax.set_xlabel('Target region')
    ax.set_ylabel('Source region')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '01_matrix_heatmaps.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: 01_matrix_heatmaps.png')

# ── figure 02: degree distributions ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('Degree Distributions (out-degree)', fontsize=14, fontweight='bold')

for ax, sp in zip(axes, SPECIES):
    w = connectomes[sp]['w']
    # out-degree = number of non-zero outgoing connections per node
    out_degree = np.sum(w > 0, axis=1)
    in_degree  = np.sum(w > 0, axis=0)
    ax.hist(out_degree, bins=20, alpha=0.7, color='#2196F3', label='Out-degree')
    ax.hist(in_degree,  bins=20, alpha=0.5, color='#FF5722', label='In-degree')
    ax.set_title(sp.upper(), fontweight='bold')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '02_degree_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 02_degree_distributions.png')
print('\nStep 01 complete.')
