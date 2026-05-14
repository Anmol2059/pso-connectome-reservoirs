"""
Download and save real biological connectomes.

Species & sources (all via netneurotools.datasets.fetch_famous_gmat):
  celegans      : Varshney et al. 2011 (C. elegans, 279 chemical-synapse regions)
  fly           : Chiang et al. 2011   (Drosophila, 49 regions)
  mouse         : Rubinov et al. 2015  (mouse cortex, 112 regions)
  rat           : Bota et al. 2015     (rat cortex, 73 regions)
  macaque_b     : Modha & Singh 2010   (macaque, 242 regions, BINARY)
  macaque_w     : Markov et al. 2014   (macaque cortex, 29 regions, FLNe weights)
  human         : Cammoun et al. 2012  (human structural MRI, Lausanne scale-033, 83 regions)

Seven species spanning C. elegans (279 neurons) to the human cortex (83 parcels),
covering three weight types: continuous chemical-synapse counts (C. elegans),
continuous FLNe axonal-tracing weights (fly, mouse, rat, macaque_w), and
structural-MRI streamline density (human).  macaque_b is the binary-weight
control used to isolate the topology vs. weight-encoding question (Sec. 4.4).

Input node assignment
  fly       : olfactory network ('olf') — biologically motivated
  celegans  : sensory neurons (mechanosensory + chemosensory, ~65 cells)
  human     : primary sensory + motor cortex parcels (V1, S1, A1 regions)
  others    : random 20% (seed=42); sensitivity to seed reported in step04

Note: macaque_w (Markov 2014) is a 93×29 rectangular dataset.
      Rows are injection sites; we extract the 29×29 square sub-matrix
      corresponding to the core areas that were both injection targets and
      received tracers.

All outputs saved as data/{species}/conn.npy + labels.npy.
"""
import os
import numpy as np
from netneurotools.datasets import fetch_famous_gmat

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

rng_global = np.random.default_rng(42)


def input_labels(n, n_input=None, input_idx=None):
    labels = np.zeros(n, dtype=int)
    if input_idx is not None:
        labels[input_idx] = 1
    else:
        k = n_input or max(1, int(0.2 * n))
        labels[rng_global.choice(n, size=k, replace=False)] = 1
    return labels


def save(sp_name, conn, labels, note=''):
    sp_dir = os.path.join(DATA_DIR, sp_name)
    os.makedirs(sp_dir, exist_ok=True)
    np.fill_diagonal(conn, 0)
    is_bin = bool(np.all((conn == 0) | (conn == 1)))
    np.save(os.path.join(sp_dir, 'conn.npy'),   conn)
    np.save(os.path.join(sp_dir, 'labels.npy'), labels)
    nnz = np.count_nonzero(conn)
    n = conn.shape[0]
    print(f'  Nodes={n}  Edges={nnz}  Density={nnz/(n*(n-1)):.4f}  '
          f'Sym={np.allclose(conn, conn.T)}  Binary={is_bin}')
    if not is_bin:
        print(f'  Weight range: [{conn[conn>0].min():.4f}, {conn.max():.4f}]')
    if note:
        print(f'  Note: {note}')
    print(f'  Saved: data/{sp_name}/')


print('=' * 65)
print('  Downloading biological connectomes via netneurotools')
print('  Seven species: C. elegans → Fly → Mouse → Rat →')
print('                 Macaque (binary) → Macaque (weighted) → Human')
print('=' * 65)

# ── 1. C. elegans ─────────────────────────────────────────────────────────────
print('\nC. ELEGANS  (Varshney et al. 2011)')
d = fetch_famous_gmat('celegans', verbose=0)
conn = d.conn.astype(float)

# Sensory neurons: mechanosensory (ALM, AVM, PLM, PVM) + chemosensory (ASE etc.)
# In the 279-neuron dataset labels are neuron names — assign sensory cells as inputs
labels_ce = d.labels  # array of neuron names
sensory_prefixes = ('ALM', 'AVM', 'PLM', 'PVM', 'ASE', 'ASG', 'ASH',
                    'ASI', 'ASJ', 'ASK', 'AWA', 'AWB', 'AWC', 'AFD',
                    'ADF', 'ADL')
if hasattr(labels_ce, '__len__') and len(labels_ce) == len(conn):
    sensory_idx = [i for i, lbl in enumerate(labels_ce)
                   if any(str(lbl).startswith(p) for p in sensory_prefixes)]
else:
    sensory_idx = None

if sensory_idx and 0 < len(sensory_idx) < len(conn):
    lbl = input_labels(len(conn), input_idx=sensory_idx)
    print(f'  Input nodes: sensory neurons ({len(sensory_idx)} cells) — biological')
else:
    lbl = input_labels(len(conn))
    print(f'  Input nodes: random 20%')

save('celegans', conn, lbl,
     note='Chemical-synapse counts, Varshney et al. 2011 PLOS CB')

# ── 2. Fly (Drosophila) ──────────────────────────────────────────────────────
print('\nFLY  (Chiang et al. 2011)')
d = fetch_famous_gmat('drosophila', verbose=0)
conn = d.conn.astype(float)
if hasattr(d, 'networks'):
    olf_idx = np.where(d.networks == 'olf')[0]
    lbl = input_labels(len(conn), input_idx=olf_idx)
    print(f'  Input nodes: olfactory network ({len(olf_idx)} nodes) — biological')
else:
    lbl = input_labels(len(conn))
save('fly', conn, lbl)

# ── 3. Mouse ─────────────────────────────────────────────────────────────────
print('\nMOUSE  (Rubinov et al. 2015)')
d = fetch_famous_gmat('mouse', verbose=0)
conn = d.conn.astype(float)
save('mouse', conn, input_labels(len(conn)))

# ── 4. Rat ───────────────────────────────────────────────────────────────────
print('\nRAT  (Bota et al. 2015)')
d = fetch_famous_gmat('rat', verbose=0)
conn = d.conn.astype(float)
save('rat', conn, input_labels(len(conn)))

# ── 5. Macaque — binary (Modha & Singh 2010) ─────────────────────────────────
print('\nMACAQUE-BINARY  (Modha & Singh 2010)')
d = fetch_famous_gmat('macaque_modha', verbose=0)
conn = d.conn.astype(float)
save('macaque_b', conn, input_labels(len(conn)),
     note='Binary adjacency only (0/1). Continuous-weight control = macaque_w.')

# ── 6. Macaque — weighted (Markov et al. 2014, FLNe) ─────────────────────────
print('\nMACAQUE-WEIGHTED  (Markov et al. 2014, FLNe)')
d = fetch_famous_gmat('macaque_markov', verbose=0)
# Dataset is rectangular: 93 injection sites × 29 target areas (FLNe values).
# We extract the square 29×29 sub-matrix where injection sites are themselves
# among the 29 core target areas (both injected and received tracer).
labels_raw = d.labels          # (93, 2): col-0 = injection, col-1 = target
injections  = labels_raw[:, 0] # 93 injection site names
targets     = np.unique(labels_raw[:, 1])
targets     = targets[targets != '']   # remove empty-string row if present

core_mask = np.isin(injections, targets)
core_idx  = np.where(core_mask)[0]
core_inj  = injections[core_mask]

# Sort both axes by the canonical target order so W[i,j] = FLNe from i→j
tgt_order = {t: k for k, t in enumerate(targets)}
sort_order = np.argsort([tgt_order[l] for l in core_inj])
core_idx   = core_idx[sort_order]
core_inj   = core_inj[sort_order]

conn_mw = d.conn[core_idx, :]   # 29×29 (rows = sorted injection areas, cols = targets)
# Remove empty column if present
valid_cols = [j for j, t in enumerate(targets) if t != '']
conn_mw = conn_mw[:, valid_cols][:len(valid_cols), :]  # ensure square
conn_mw = conn_mw.astype(float)

save('macaque_w', conn_mw, input_labels(len(conn_mw)),
     note='FLNe continuous weights, Markov et al. 2014 Cerebral Cortex. '
          '29 core cortical areas. Direct weighted comparison with macaque_b.')

# ── 7. Human — structural MRI (Cammoun et al. 2012, scale-033) ──────────────
print('\nHUMAN  (Cammoun et al. 2012, Lausanne scale-033, 83 parcels)')
d = fetch_famous_gmat('human_struct_scale033', verbose=0)
conn = d.conn.astype(float)

# Primary sensory / motor parcels in the Lausanne atlas at scale-033:
# Approximate — labels are Lausanne ROI names; use left hemisphere if available
if hasattr(d, 'labels') and len(d.labels) == len(conn):
    sensorimotor = ('ctx-lh-precentral', 'ctx-rh-precentral',
                    'ctx-lh-postcentral', 'ctx-rh-postcentral',
                    'ctx-lh-pericalcarine', 'ctx-rh-pericalcarine',
                    'ctx-lh-transversetemporal', 'ctx-rh-transversetemporal')
    sm_idx = [i for i, lbl in enumerate(d.labels)
              if any(s in str(lbl).lower() for s in
                     ('precentral', 'postcentral', 'pericalcarine', 'transverse'))]
    if 0 < len(sm_idx) < len(conn):
        lbl = input_labels(len(conn), input_idx=sm_idx)
        print(f'  Input nodes: primary sensorimotor/visual ({len(sm_idx)} parcels)')
    else:
        lbl = input_labels(len(conn))
        print('  Input nodes: random 20%')
else:
    lbl = input_labels(len(conn))
    print('  Input nodes: random 20%')

save('human', conn, lbl,
     note='Structural MRI streamline density, Lausanne scale-033 (83 parcels). '
          'Symmetric/undirected (diffusion MRI); differs from axonal-tracing species.')

print('\n' + '=' * 65)
print('  All 7 connectomes saved.')
print('  Species by network size:')
for sp in ('celegans', 'fly', 'mouse', 'rat', 'macaque_b', 'macaque_w', 'human'):
    c = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    n = c.shape[0]
    lbl = np.load(os.path.join(DATA_DIR, sp, 'labels.npy'))
    print(f'    {sp:15s}  N={n:3d}  inputs={lbl.sum():3d}')
print('=' * 65)
