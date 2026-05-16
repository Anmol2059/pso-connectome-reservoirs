"""
Step 02 — Baseline Tasks: Memory Capacity + Lorenz Prediction
=============================================================
Evaluates biological connectomes on two reservoir computing benchmarks.

Scientific fixes applied (Reviewer 2):
  - ALPHA = 0.97 (was 0.9): MC increases monotonically as rho→1; 0.95-0.99 is
    standard for MC benchmarks (Dambre et al. 2012, Verstraeten et al. 2010)
  - RidgeCV with alpha in {0.01, 0.1, 1.0, 10, 100}: cross-validated readout;
    alpha=1e-6 is essentially unregularised and can overfit (#6)
  - WASHOUT = 200 steps for proper transient elimination
  - MAX_LAG  = 50        for richer MC profile
  - N_RUNS   = 5         for statistical validity; Cohen's d reported alongside
    t-test results to characterise effect size at low df (#4)
  - Each run uses a different random signal seed → independent trials

Outputs:
  data/baseline_results.csv  — columns: species, run, mc, lorenz_nrmse
  images/03_mc_baselines.png
  images/04_lorenz_baselines.png
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from scipy.integrate import solve_ivp
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── paths ─────────────────────────────────────────────────────────────────────
PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(PROJ_DIR, 'data')
IMAGES_DIR = os.path.join(PROJ_DIR, 'images')
CONN2RES   = os.path.join(os.path.dirname(PROJ_DIR), 'conn2res')
if os.path.exists(CONN2RES) and CONN2RES not in sys.path:
    sys.path.insert(0, CONN2RES)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
SPECIES    = ['celegans', 'fly', 'mouse', 'rat', 'macaque_w', 'human']
N_RUNS     = 5       # independent evaluations per network
N_SAMPLES  = 2000    # time steps per evaluation (after washout)
MAX_LAG    = 50      # MC lags; extended from 20 → captures longer-range memory
WASHOUT    = 200     # discard first 200 ESN states; standard ESN practice
ALPHA      = 0.97    # spectral radius — closer to 1 maximises MC (Dambre 2012)
TRAIN_FRAC = 0.7

# Cross-validated ridge alphas — prevents overfitting at ~80% output nodes (≈193 for macaque)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# ── fast spectral radius ──────────────────────────────────────────────────────
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import eigs as sparse_eigs

def spectral_radius(w):
    """Largest absolute eigenvalue via sparse Arnoldi. Falls back to dense."""
    try:
        ev = sparse_eigs(sp_sparse.csr_matrix(w), k=1, which='LM',
                         return_eigenvectors=False, maxiter=500, tol=1e-4)
        return float(np.abs(ev[0]))
    except Exception:
        return float(np.max(np.abs(np.linalg.eigvals(w))))


def sim_esn(w_r, u_in_proj, output_nodes, washout):
    """
    w_r         : (n, n) reservoir weights, already spectral-radius-scaled
    u_in_proj   : (washout + n_samples, n) pre-computed input projections
    output_nodes: indices of readout nodes
    washout     : number of leading states to discard
    Returns reservoir states[washout:] for output_nodes only.
    """
    T_total = len(u_in_proj)
    n       = w_r.shape[0]
    s       = np.zeros(n)
    T_use   = T_total - washout
    out     = np.empty((T_use, len(output_nodes)))
    for t in range(T_total):
        s = np.tanh(s @ w_r + u_in_proj[t])
        if t >= washout:
            out[t - washout] = s[output_nodes]
    return out


# ── memory capacity ───────────────────────────────────────────────────────────
def mc_score(w_bio, w_in, output_nodes, seed=0):
    """Memory Capacity for weight matrix w_bio with a given random seed."""
    try:
        sr = spectral_radius(w_bio)
        if sr < 1e-10:
            return 0.0, None
        w_r = (ALPHA / sr) * w_bio

        rng     = np.random.default_rng(seed)
        n_total = WASHOUT + N_SAMPLES + MAX_LAG
        u       = rng.uniform(-1, 1, size=n_total)

        x_full = u[MAX_LAG:].reshape(-1, 1)
        y_full = np.column_stack([
            u[MAX_LAG - k: MAX_LAG - k + WASHOUT + N_SAMPLES]
            for k in range(1, MAX_LAG + 1)
        ])

        u_proj = x_full @ w_in
        rs     = sim_esn(w_r, u_proj, output_nodes, WASHOUT)
        y      = y_full[WASHOUT:]

        n_train = int(TRAIN_FRAC * N_SAMPLES)
        model   = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
        model.fit(rs[:n_train], y[:n_train])
        y_pred  = model.predict(rs[n_train:])

        score = 0.0
        for k in range(MAX_LAG):
            yt, yp = y[n_train:, k], y_pred[:, k]
            if np.std(yp) > 1e-10 and np.std(yt) > 1e-10:
                score += float(np.corrcoef(yt, yp)[0, 1] ** 2)
        return score, getattr(model, 'alpha_', None)
    except Exception:
        return 0.0, None


# ── lorenz prediction ─────────────────────────────────────────────────────────
def lorenz_ode(t, s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    return [sigma * (s[1] - s[0]),
            s[0] * (rho - s[2]) - s[1],
            s[0] * s[1] - beta * s[2]]


def lorenz_score(w_bio, w_in, output_nodes, seed=0):
    """One-step-ahead Lorenz prediction; returns NRMSE (lower = better)."""
    try:
        sr = spectral_radius(w_bio)
        if sr < 1e-10:
            return 1.0
        w_r = (ALPHA / sr) * w_bio

        rng = np.random.default_rng(seed)
        ic  = [10.0 + rng.uniform(-1, 1),
               10.0 + rng.uniform(-1, 1),
               28.0 + rng.uniform(-1, 1)]
        t_end = (WASHOUT + N_SAMPLES + 1) * 0.02
        sol   = solve_ivp(lorenz_ode, [0, t_end], ic,
                          t_eval=np.arange(0, t_end, 0.02), dense_output=False)
        sig   = sol.y[0]
        sig   = (sig - sig.mean()) / (sig.std() + 1e-10)
        sig   = sig[:WASHOUT + N_SAMPLES + 1]

        x_full = sig[:WASHOUT + N_SAMPLES].reshape(-1, 1)
        y_full = sig[1:WASHOUT + N_SAMPLES + 1]

        u_proj = x_full @ w_in
        rs     = sim_esn(w_r, u_proj, output_nodes, WASHOUT)
        y_use  = y_full[WASHOUT:]

        n_train = int(TRAIN_FRAC * N_SAMPLES)
        model   = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
        model.fit(rs[:n_train], y_use[:n_train])
        y_pred  = model.predict(rs[n_train:]).flatten()
        y_test  = y_use[n_train:]

        nrmse = np.sqrt(np.mean((y_test - y_pred) ** 2)) / (
            y_test.max() - y_test.min() + 1e-10)
        return float(nrmse)
    except Exception:
        return 1.0


# ── main ──────────────────────────────────────────────────────────────────────
print(f'Config: N_RUNS={N_RUNS}, N_SAMPLES={N_SAMPLES}, '
      f'MAX_LAG={MAX_LAG}, WASHOUT={WASHOUT}, ALPHA={ALPHA}')
print(f'Readout: RidgeCV(alphas={RIDGE_ALPHAS}) with 5-fold CV')
print()

t0_total = time.time()
rows = []

for sp in SPECIES:
    t0_sp = time.time()
    w      = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    labels = np.load(os.path.join(DATA_DIR, sp, 'labels.npy'))
    n      = w.shape[0]

    is_binary = bool(np.all((w == 0) | (w == 1)))
    input_nodes  = np.where(labels == 1)[0]
    output_nodes = np.where(labels == 0)[0]
    w_in         = np.zeros((1, n))
    w_in[:, input_nodes] = 1.0 / max(1, len(input_nodes))

    print(f'\n{sp.upper()}  ({n} nodes | '
          f'{"binary" if is_binary else "weighted"} connectome)')

    mc_scores, lorenz_scores, alphas_used = [], [], []

    for run in tqdm(range(N_RUNS), desc=f'{sp} baselines'):
        mc, alpha_cv = mc_score(w, w_in, output_nodes, seed=run)
        lz = lorenz_score(w, w_in, output_nodes, seed=run)
        mc_scores.append(mc)
        lorenz_scores.append(lz)
        if alpha_cv is not None:
            alphas_used.append(alpha_cv)
        rows.append({'species': sp, 'run': run, 'mc': mc, 'lorenz_nrmse': lz})

    elapsed = time.time() - t0_sp
    print(f'  MC:     {np.mean(mc_scores):.3f} ± {np.std(mc_scores):.3f}')
    print(f'  Lorenz: {np.mean(lorenz_scores):.4f} ± {np.std(lorenz_scores):.4f}')
    if alphas_used:
        from collections import Counter
        cnt = Counter(alphas_used)
        print(f'  CV-selected alphas: {dict(cnt)}')
    print(f'  Time: {elapsed:.1f}s')

# ── save CSV ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, 'baseline_results.csv'), index=False)
print(f'\nSaved: baseline_results.csv  (total time: {time.time()-t0_total:.0f}s)')

# ── figure 03: MC bar + error bars ────────────────────────────────────────────
summary = df.groupby('species').agg(
    mc_mean=('mc', 'mean'), mc_std=('mc', 'std'),
    lorenz_mean=('lorenz_nrmse', 'mean'), lorenz_std=('lorenz_nrmse', 'std')
).reindex(SPECIES)

x   = np.arange(len(SPECIES))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x, summary['mc_mean'], yerr=summary['mc_std'], capsize=5,
       color='#2196F3', alpha=0.85, edgecolor='white', error_kw={'linewidth': 1.5})
ax.set_xticks(x)
ax.set_xticklabels([s.upper() for s in SPECIES], fontsize=12)
ax.set_ylabel(f'Memory Capacity (Σ R², {MAX_LAG} lags)', fontsize=12)
ax.set_title(f'Biological Connectome — Memory Capacity Baseline\n'
             f'(mean ± SD, N={N_RUNS} independent signal seeds, ρ={ALPHA})',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '03_mc_baselines.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 03_mc_baselines.png')

# ── figure 04: Lorenz bar + error bars ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x, summary['lorenz_mean'], yerr=summary['lorenz_std'], capsize=5,
       color='#FF5722', alpha=0.85, edgecolor='white', error_kw={'linewidth': 1.5})
ax.set_xticks(x)
ax.set_xticklabels([s.upper() for s in SPECIES], fontsize=12)
ax.set_ylabel('NRMSE (lower = better)', fontsize=12)
ax.set_title(f'Biological Connectome — Lorenz Prediction Baseline\n'
             f'(mean ± SD, N={N_RUNS} independent initial conditions, ρ={ALPHA})',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '04_lorenz_baselines.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 04_lorenz_baselines.png')

print('\nStep 02 complete.')
