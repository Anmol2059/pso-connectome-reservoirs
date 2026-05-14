"""
Step 03 — Multi-Algorithm Bio-Inspired Optimisation of Connectome Reservoirs
=============================================================================
Optimises biological connectome edge weights with three bio-inspired algorithms:

  PSO  — Particle Swarm Optimisation   (Kennedy & Eberhart 1995)
           Inspired by collective movement of birds/fish schools.
  DE   — Differential Evolution        (Storn & Price 1997)
           Inspired by Darwinian genetic recombination and mutation.
  GWO  — Grey Wolf Optimiser           (Mirjalili et al. 2014)
           Inspired by grey wolf pack hunting hierarchy (alpha/beta/delta).

All three are gradient-free, bio-inspired, and population-based.
They are compared on two reservoir computing tasks:
  MC      — Memory Capacity (sum of squared correlations up to MAX_LAG)
  Lorenz  — Lorenz attractor NRMSE (one-step-ahead prediction)

Conditions per species:
  Bio-A   — Optimiser initialised near biological weights (Gaussian perturbation)
  Rand-B  — PSO only, initialised from uniform random weights (negative control)
  Random  — Random re-weighting, no optimisation (null baseline)
  Bio     — Unoptimised biological weights (held-out evaluation)

Scientific design:
  Held-out evaluation: optimisers search on signal A (seed=run),
  all reported scores use signal B (seed=run+EVAL_SEED_OFFSET).
  PSO/DE/GWO all use the same GPU-batched objective for fair comparison.
  All three use identical populations (N_PARTICLES=20, N_ITER=50).

Outputs:
  data/opt_results.csv              — all conditions × algorithms × tasks × runs
  data/{sp}/conn_{alg}_a.npy        — best weights per species/algorithm (MC task)
  images/05_convergence_mc.png      — convergence curves (MC task)
  images/05_convergence_lorenz.png  — convergence curves (Lorenz task)
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from tqdm import tqdm
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import eigs as sparse_eigs
from pyswarms.single import GlobalBestPSO

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── GPU / device ──────────────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
    USE_GPU = DEVICE.type != 'cpu'
except ImportError:
    torch   = None
    DEVICE  = None
    USE_GPU = False

print(f'[device] {DEVICE}  {"← GPU batch active" if USE_GPU else "← CPU"}')

# ── paths ──────────────────────────────────────────────────────────────────────
PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(PROJ_DIR, 'data')
IMAGES_DIR = os.path.join(PROJ_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
SPECIES = ['celegans', 'fly', 'mouse', 'rat', 'macaque_b', 'macaque_w', 'human']

N_RUNS       = 5
N_SAMPLES    = 2000
MAX_LAG      = 50
WASHOUT      = 200
ALPHA        = 0.97    # spectral radius target (Dambre et al. 2012)
TRAIN_FRAC   = 0.7
MAX_PERTURB  = 0.3     # std scaling for Bio-A initialisation

EVAL_SEED_OFFSET = 100  # separates optimisation signal from held-out signal

N_PARTICLES  = 20
N_ITER       = 50      # 20×50 = 1000 evaluations per run

PSO_OPTIONS  = {'c1': 2.0, 'c2': 2.0, 'w': 0.7}
DE_F         = 0.8     # mutation factor
DE_CR        = 0.9     # crossover rate
WEIGHT_SCALE = 3.0     # search bound = bio_max × WEIGHT_SCALE
RIDGE_ALPHA  = 1.0     # fixed Ridge in objective (CV too slow in inner loop)

TASKS = ['mc', 'lorenz']
ALGORITHMS = ['pso', 'de', 'gwo']

print(f'Config: {N_RUNS} runs × {N_PARTICLES} particles × {N_ITER} iter '
      f'= {N_PARTICLES*N_ITER} evals/run')
print(f'Tasks: {TASKS}   Algorithms: {ALGORITHMS}')
print(f'Species: {SPECIES}')
print(f'Held-out eval seed offset = {EVAL_SEED_OFFSET}')

# ── Lorenz attractor ───────────────────────────────────────────────────────────
def lorenz_trajectory(n_steps, sigma=10., rho=28., beta=8./3., dt=0.02, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, 3)
    traj = np.empty((n_steps, 3))
    for i in range(n_steps):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        x = x + dt * np.array([dx, dy, dz])
        traj[i] = x
    return traj


def make_lorenz_data(seed):
    n_total = WASHOUT + N_SAMPLES + 1
    traj = lorenz_trajectory(n_total, seed=seed)
    x_full = traj[:, 0:1]          # use x-coordinate as input
    x_in   = x_full[:-1]           # input at t
    y_out  = x_full[1:, 0]         # predict x at t+1
    # normalise
    mu, std = x_in.mean(), x_in.std() + 1e-10
    x_in = (x_in - mu) / std
    y_out = (y_out - mu) / std
    return x_in, y_out             # shapes (WASHOUT+N_SAMPLES, 1) and (WASHOUT+N_SAMPLES,)


# ── MC data ────────────────────────────────────────────────────────────────────
def make_mc_data(seed):
    rng     = np.random.default_rng(seed)
    n_total = WASHOUT + N_SAMPLES + MAX_LAG
    u       = rng.uniform(-1, 1, size=n_total)
    x_full  = u[MAX_LAG:].reshape(-1, 1)
    y_full  = np.column_stack([
        u[MAX_LAG - k: MAX_LAG - k + WASHOUT + N_SAMPLES]
        for k in range(1, MAX_LAG + 1)
    ])[WASHOUT:]
    return x_full, y_full


# ── spectral radius ────────────────────────────────────────────────────────────
def spectral_radius(w):
    try:
        ev = sparse_eigs(sp_sparse.csr_matrix(w), k=1, which='LM',
                         return_eigenvectors=False, maxiter=500, tol=1e-4)
        return float(np.abs(ev[0]))
    except Exception:
        return float(np.max(np.abs(np.linalg.eigvals(w))))


# ── CPU ESN simulation ─────────────────────────────────────────────────────────
def sim_esn(w_r, u_proj, output_nodes, washout):
    T_total = len(u_proj)
    n       = w_r.shape[0]
    s       = np.zeros(n)
    T_use   = T_total - washout
    out     = np.empty((T_use, len(output_nodes)))
    for t in range(T_total):
        s = np.tanh(s @ w_r + u_proj[t])
        if t >= washout:
            out[t - washout] = s[output_nodes]
    return out


# ── Ridge MC score ─────────────────────────────────────────────────────────────
def ridge_mc(rs, y_tgt, n_train):
    try:
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(rs[:n_train], y_tgt[:n_train])
        y_pred = m.predict(rs[n_train:])
        score  = sum(
            float(np.corrcoef(y_tgt[n_train:, k], y_pred[:, k])[0, 1] ** 2)
            for k in range(MAX_LAG)
            if np.std(y_pred[:, k]) > 1e-10 and np.std(y_tgt[n_train:, k]) > 1e-10
        )
        return score
    except Exception:
        return 0.0


def ridge_lorenz_nrmse(rs, y_tgt, n_train):
    try:
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(rs[:n_train], y_tgt[:n_train])
        y_pred = m.predict(rs[n_train:])
        nrmse  = np.sqrt(np.mean((y_tgt[n_train:] - y_pred) ** 2))
        nrmse /= (np.std(y_tgt[n_train:]) + 1e-10)
        return float(nrmse)
    except Exception:
        return 1.0


# ── single-weight-matrix scorers ──────────────────────────────────────────────
def score_mc(w_raw, w_in, output_nodes, u_proj, y_tgt):
    try:
        sr = spectral_radius(w_raw)
        if sr < 1e-10: return 0.0
        w_r     = (ALPHA / sr) * w_raw
        rs      = sim_esn(w_r, u_proj, output_nodes, WASHOUT)
        n_train = int(TRAIN_FRAC * N_SAMPLES)
        return ridge_mc(rs, y_tgt, n_train)
    except Exception:
        return 0.0


def score_lorenz(w_raw, w_in, output_nodes, u_proj, y_tgt):
    try:
        sr = spectral_radius(w_raw)
        if sr < 1e-10: return 1.0
        w_r     = (ALPHA / sr) * w_raw
        rs      = sim_esn(w_r, u_proj, output_nodes, WASHOUT)
        n_train = int(TRAIN_FRAC * N_SAMPLES)
        return ridge_lorenz_nrmse(rs, y_tgt, n_train)
    except Exception:
        return 1.0


# ── GPU batch infrastructure ──────────────────────────────────────────────────
def _batch_spectral_radius(W_batch, n_iter=50):
    P, n, _ = W_batch.shape
    v = torch.randn(P, n, 1, device=W_batch.device, dtype=W_batch.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-10)
    for _ in range(n_iter):
        v = torch.bmm(W_batch, v)
        v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return torch.bmm(W_batch, v).norm(dim=1).squeeze(-1)


def _batch_sim_esn(W_batch, u_proj_t, output_nodes, washout):
    P, n, _  = W_batch.shape
    T_total  = len(u_proj_t)
    T_use    = T_total - washout
    n_out    = len(output_nodes)
    s   = torch.zeros(P, n, device=W_batch.device, dtype=W_batch.dtype)
    out = torch.empty(P, T_use, n_out, device=W_batch.device, dtype=W_batch.dtype)
    for t in range(T_total):
        s = torch.tanh(torch.bmm(s.unsqueeze(1), W_batch).squeeze(1) + u_proj_t[t])
        if t >= washout:
            out[:, t - washout, :] = s[:, output_nodes]
    return out.cpu().numpy()


def make_batch_objective_mc(n, nz_rows, nz_cols, w_in, output_nodes, u_proj, y_tgt):
    """GPU-batched objective: returns minimisation cost (-MC) for each particle."""
    if not USE_GPU:
        def cpu_obj(particles):
            costs = []
            for p in particles:
                w = np.zeros((n, n)); w[nz_rows, nz_cols] = np.abs(p)
                costs.append(-score_mc(w, w_in, output_nodes, u_proj, y_tgt))
            return np.array(costs)
        return cpu_obj

    nz_r_t  = torch.tensor(nz_rows, device=DEVICE, dtype=torch.long)
    nz_c_t  = torch.tensor(nz_cols, device=DEVICE, dtype=torch.long)
    u_t     = torch.from_numpy(u_proj.astype(np.float32)).to(DEVICE)
    n_train = int(TRAIN_FRAC * N_SAMPLES)
    out_idx = list(output_nodes)

    def gpu_obj(particles):
        P     = len(particles)
        parts = torch.from_numpy(np.abs(particles).astype(np.float32)).to(DEVICE)
        W_raw = torch.zeros(P, n, n, device=DEVICE, dtype=torch.float32)
        W_raw[:, nz_r_t, nz_c_t] = parts
        sr       = _batch_spectral_radius(W_raw).clamp(min=1e-10)
        W_scaled = W_raw * (ALPHA / sr).view(P, 1, 1)
        rs_np    = _batch_sim_esn(W_scaled, u_t, out_idx, WASHOUT)
        return np.array([-ridge_mc(rs_np[i], y_tgt, n_train) for i in range(P)])

    return gpu_obj


def make_batch_objective_lorenz(n, nz_rows, nz_cols, w_in, output_nodes, u_proj, y_tgt):
    """GPU-batched Lorenz objective: returns NRMSE for each particle."""
    if not USE_GPU:
        def cpu_obj(particles):
            costs = []
            for p in particles:
                w = np.zeros((n, n)); w[nz_rows, nz_cols] = np.abs(p)
                costs.append(score_lorenz(w, w_in, output_nodes, u_proj, y_tgt))
            return np.array(costs)
        return cpu_obj

    nz_r_t  = torch.tensor(nz_rows, device=DEVICE, dtype=torch.long)
    nz_c_t  = torch.tensor(nz_cols, device=DEVICE, dtype=torch.long)
    u_t     = torch.from_numpy(u_proj.astype(np.float32)).to(DEVICE)
    n_train = int(TRAIN_FRAC * N_SAMPLES)
    out_idx = list(output_nodes)

    def gpu_obj(particles):
        P     = len(particles)
        parts = torch.from_numpy(np.abs(particles).astype(np.float32)).to(DEVICE)
        W_raw = torch.zeros(P, n, n, device=DEVICE, dtype=torch.float32)
        W_raw[:, nz_r_t, nz_c_t] = parts
        sr       = _batch_spectral_radius(W_raw).clamp(min=1e-10)
        W_scaled = W_raw * (ALPHA / sr).view(P, 1, 1)
        rs_np    = _batch_sim_esn(W_scaled, u_t, out_idx, WASHOUT)
        return np.array([ridge_lorenz_nrmse(rs_np[i], y_tgt, n_train) for i in range(P)])

    return gpu_obj


def weights_from_particle(p, n, nz_rows, nz_cols):
    w = np.zeros((n, n))
    w[nz_rows, nz_cols] = np.abs(p)
    return w


# ── PSO ───────────────────────────────────────────────────────────────────────
def run_pso(objective, bounds_lb, bounds_ub, n_particles, n_iter, init_pos, label):
    """Wrapper around pyswarms GlobalBestPSO with tqdm history tracking."""
    n_dims  = len(bounds_lb)
    history = []

    class _Obj:
        def __call__(self, particles):
            costs = objective(particles)
            history.append(float(-costs.min()))
            return costs

    wrapped = _Obj()
    opt = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_dims,
        options=PSO_OPTIONS,
        bounds=(bounds_lb, bounds_ub),
        init_pos=init_pos,
    )
    pbar = tqdm(total=n_iter, desc=label, unit='iter', leave=False)
    original_call = wrapped.__call__

    def _tracked(particles):
        costs = original_call(particles)
        pbar.set_postfix(best=f'{history[-1]:.3f}')
        pbar.update(1)
        return costs

    wrapped.__call__ = _tracked
    best_cost, best_pos = opt.optimize(wrapped, iters=n_iter, verbose=False)
    pbar.close()
    return best_pos, -best_cost, history


# ── DE (Storn & Price 1997) ───────────────────────────────────────────────────
def run_de(objective, bounds_lb, bounds_ub, n_particles, n_iter, init_pos, label,
           F=DE_F, CR=DE_CR, seed=0):
    """
    Vectorised DE/rand/1/bin using the shared GPU-batched objective.
    All trial vectors are evaluated in a single batch call per iteration.
    Storn R & Price KV (1997) J Global Optim 11:341-359.
    """
    n_dims = len(bounds_lb)
    rng    = np.random.default_rng(seed)
    pop    = np.clip(init_pos.copy(), bounds_lb, bounds_ub)
    scores = objective(pop)
    history = []

    pbar = tqdm(total=n_iter, desc=label, unit='iter', leave=False)
    for t in range(n_iter):
        trials = np.empty_like(pop)
        for i in range(n_particles):
            cands  = [j for j in range(n_particles) if j != i]
            a, b, c = rng.choice(cands, 3, replace=False)
            mutant  = np.clip(pop[a] + F * (pop[b] - pop[c]), bounds_lb, bounds_ub)
            mask    = rng.random(n_dims) <= CR
            if not mask.any():
                mask[rng.integers(n_dims)] = True
            trials[i] = np.where(mask, mutant, pop[i])

        trial_scores = objective(trials)          # single GPU batch call
        improved     = trial_scores < scores
        pop[improved]    = trials[improved]
        scores[improved] = trial_scores[improved]

        best_mc = -scores.min()
        history.append(best_mc)
        pbar.set_postfix(best=f'{best_mc:.3f}')
        pbar.update(1)
    pbar.close()

    best_idx = scores.argmin()
    return pop[best_idx], -scores[best_idx], history


# ── GWO (Mirjalili et al. 2014) ──────────────────────────────────────────────
def run_gwo(objective, bounds_lb, bounds_ub, n_wolves, n_iter, init_pos, label, seed=0):
    """
    Vectorised Grey Wolf Optimiser using GPU-batched objective.
    All wolves are updated and evaluated as a batch per iteration.
    Mirjalili S et al. (2014) Adv Eng Softw 69:46-61.
    """
    n_dims = len(bounds_lb)
    rng    = np.random.default_rng(seed)
    wolves = np.clip(init_pos.copy(), bounds_lb, bounds_ub)
    scores = objective(wolves)

    sorted_idx  = np.argsort(scores)
    alpha_pos   = wolves[sorted_idx[0]].copy()
    beta_pos    = wolves[sorted_idx[1]].copy()
    delta_pos   = wolves[sorted_idx[2]].copy()
    alpha_score = scores[sorted_idx[0]]

    history = []
    pbar = tqdm(total=n_iter, desc=label, unit='iter', leave=False)

    for t in range(n_iter):
        a = 2.0 * (1.0 - t / n_iter)   # linearly 2 → 0

        r1 = rng.random((n_wolves, n_dims))
        r2 = rng.random((n_wolves, n_dims))
        X1 = alpha_pos - (2*a*r1 - a) * np.abs(2*r2*alpha_pos - wolves)

        r1 = rng.random((n_wolves, n_dims))
        r2 = rng.random((n_wolves, n_dims))
        X2 = beta_pos  - (2*a*r1 - a) * np.abs(2*r2*beta_pos  - wolves)

        r1 = rng.random((n_wolves, n_dims))
        r2 = rng.random((n_wolves, n_dims))
        X3 = delta_pos - (2*a*r1 - a) * np.abs(2*r2*delta_pos - wolves)

        wolves = np.clip((X1 + X2 + X3) / 3.0, bounds_lb, bounds_ub)
        scores = objective(wolves)          # single GPU batch call

        si = np.argsort(scores)
        if scores[si[0]] < alpha_score:
            alpha_score = scores[si[0]]
            alpha_pos   = wolves[si[0]].copy()
            beta_pos    = wolves[si[1]].copy()
            delta_pos   = wolves[si[2]].copy()

        best_mc = -alpha_score
        history.append(best_mc)
        pbar.set_postfix(best=f'{best_mc:.3f}')
        pbar.update(1)
    pbar.close()

    return alpha_pos, -alpha_score, history


OPTIMIZER_FN = {'pso': run_pso, 'de': run_de, 'gwo': run_gwo}


# ── main loop ─────────────────────────────────────────────────────────────────
all_rows     = []
conv_mc      = {sp: {alg: [] for alg in ALGORITHMS} for sp in SPECIES}
conv_lorenz  = {sp: {alg: [] for alg in ALGORITHMS} for sp in SPECIES}
t0_total     = time.time()

for sp in SPECIES:
    t0_sp  = time.time()
    w_bio  = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    labels = np.load(os.path.join(DATA_DIR, sp, 'labels.npy'))
    n      = w_bio.shape[0]

    is_binary      = bool(np.all((w_bio == 0) | (w_bio == 1)))
    nz_rows, nz_cols = np.where(w_bio > 0)
    n_weights      = len(nz_rows)
    bio_weights    = w_bio[nz_rows, nz_cols]
    max_w          = bio_weights.max() * WEIGHT_SCALE
    bounds_lb      = np.zeros(n_weights)
    bounds_ub      = np.full(n_weights, max_w)

    input_nodes    = np.where(labels == 1)[0]
    output_nodes   = np.where(labels == 0)[0]
    w_in           = np.zeros((1, n))
    w_in[:, input_nodes] = 1.0 / max(1, len(input_nodes))

    print(f'\n{"="*65}')
    print(f'SPECIES: {sp.upper()}  | {n} nodes | {n_weights} edges | '
          f'{"BINARY" if is_binary else f"max_w={max_w:.3f}"}')

    for run in range(N_RUNS):
        print(f'\n  Run {run+1}/{N_RUNS}')
        init_rng = np.random.default_rng(run * 997 + 31)

        # Separate optimisation and held-out evaluation signals
        x_mc_opt,  y_mc_opt  = make_mc_data(seed=run)
        x_mc_eval, y_mc_eval = make_mc_data(seed=run + EVAL_SEED_OFFSET)
        x_lz_opt,  y_lz_opt  = make_lorenz_data(seed=run)
        x_lz_eval, y_lz_eval = make_lorenz_data(seed=run + EVAL_SEED_OFFSET)

        u_mc_opt  = x_mc_opt  @ w_in
        u_mc_eval = x_mc_eval @ w_in
        u_lz_opt  = x_lz_opt  @ w_in
        u_lz_eval = x_lz_eval @ w_in

        # Bio-A initialisation (perturbed biological weights)
        noise  = init_rng.normal(0, MAX_PERTURB * bio_weights.std(),
                                 size=(N_PARTICLES, n_weights))
        init_a = np.clip(bio_weights[np.newaxis, :] + noise, 0, max_w)
        # Rand-B initialisation (uniform random)
        init_b = init_rng.uniform(0, max_w, size=(N_PARTICLES, n_weights))

        # Shared GPU objectives per task
        obj_mc_opt   = make_batch_objective_mc(
            n, nz_rows, nz_cols, w_in, output_nodes, u_mc_opt,  y_mc_opt)
        obj_lz_opt   = make_batch_objective_lorenz(
            n, nz_rows, nz_cols, w_in, output_nodes, u_lz_opt,  y_lz_opt)

        row = {'species': sp, 'run': run,
               'n_nodes': n, 'n_edges': n_weights, 'is_binary': is_binary}

        # Bio (unoptimised) on held-out signals
        w_bio_eval = w_bio.copy()
        row['mc_bio']     = score_mc(w_bio_eval, w_in, output_nodes, u_mc_eval, y_mc_eval)
        row['lorenz_bio'] = score_lorenz(w_bio_eval, w_in, output_nodes, u_lz_eval, y_lz_eval)

        # Random null
        w_rand = np.zeros((n, n))
        w_rand[nz_rows, nz_cols] = init_rng.uniform(0, max_w, n_weights)
        row['mc_random']     = score_mc(w_rand, w_in, output_nodes, u_mc_eval, y_mc_eval)
        row['lorenz_random'] = score_lorenz(w_rand, w_in, output_nodes, u_lz_eval, y_lz_eval)

        # PSO-B (random init, MC task only — negative control)
        seed_b = run * 13 + 7
        best_b, score_b_mc, _ = OPTIMIZER_FN['pso'](
            obj_mc_opt, bounds_lb, bounds_ub, N_PARTICLES, N_ITER,
            init_b.copy(), f'{sp.upper()} PSO-B MC r{run}')
        w_best_b = weights_from_particle(best_b, n, nz_rows, nz_cols)
        row['mc_pso_b'] = score_mc(w_best_b, w_in, output_nodes, u_mc_eval, y_mc_eval)

        # Bio-A: all 3 algorithms × 2 tasks
        for alg in ALGORITHMS:
            fn = OPTIMIZER_FN[alg]
            alg_seed = run * 997 + 31 + hash(alg) % 100

            # MC optimisation
            lbl_mc = f'{sp.upper()} {alg.upper()}-A MC r{run}'
            best_mc_pos, _, hist_mc = fn(
                obj_mc_opt, bounds_lb, bounds_ub, N_PARTICLES, N_ITER,
                init_a.copy(), lbl_mc, **({'seed': alg_seed} if alg != 'pso' else {}))
            w_mc = weights_from_particle(best_mc_pos, n, nz_rows, nz_cols)
            row[f'mc_{alg}_a']  = score_mc(w_mc, w_in, output_nodes, u_mc_eval, y_mc_eval)
            conv_mc[sp][alg].append(hist_mc)

            # Lorenz optimisation
            lbl_lz = f'{sp.upper()} {alg.upper()}-A Lorenz r{run}'
            best_lz_pos, _, hist_lz = fn(
                obj_lz_opt, bounds_lb, bounds_ub, N_PARTICLES, N_ITER,
                init_a.copy(), lbl_lz, **({'seed': alg_seed + 1} if alg != 'pso' else {}))
            w_lz = weights_from_particle(best_lz_pos, n, nz_rows, nz_cols)
            row[f'lorenz_{alg}_a'] = score_lorenz(w_lz, w_in, output_nodes, u_lz_eval, y_lz_eval)
            conv_lorenz[sp][alg].append(hist_lz)

            # Save best weights (MC task, first run)
            if run == 0:
                sp_dir = os.path.join(DATA_DIR, sp)
                np.save(os.path.join(sp_dir, f'conn_{alg}_a.npy'), w_mc)

        all_rows.append(row)

        print(f'    MC:  bio={row["mc_bio"]:.3f}  '
              f'pso={row.get("mc_pso_a"):.3f}  '
              f'de={row.get("mc_de_a"):.3f}  '
              f'gwo={row.get("mc_gwo_a"):.3f}  '
              f'rand={row["mc_random"]:.3f}')
        print(f'    Lz:  bio={row["lorenz_bio"]:.4f}  '
              f'pso={row.get("lorenz_pso_a"):.4f}  '
              f'de={row.get("lorenz_de_a"):.4f}  '
              f'gwo={row.get("lorenz_gwo_a"):.4f}  '
              f'rand={row["lorenz_random"]:.4f}')

    elapsed = time.time() - t0_sp
    print(f'\n  {sp.upper()} done ({elapsed:.0f}s = {elapsed/60:.1f} min)')

# ── save results ───────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(DATA_DIR, 'opt_results.csv'), index=False)
print(f'\nSaved: opt_results.csv  ({len(df)} rows)')

# ── convergence plots ──────────────────────────────────────────────────────────
COLORS = {'pso': '#4CAF50', 'de': '#FF9800', 'gwo': '#9C27B0'}
ALG_LABELS = {'pso': 'PSO', 'de': 'DE', 'gwo': 'GWO'}

for task_name, conv_data in [('mc', conv_mc), ('lorenz', conv_lorenz)]:
    ylab = 'MC Score' if task_name == 'mc' else 'NRMSE (lower = better)'
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    fig.suptitle(
        f'Bio-Inspired Optimiser Convergence — {task_name.upper()} Task\n'
        f'All {N_RUNS} runs (light) + best run highlighted  '
        f'({N_PARTICLES} particles × {N_ITER} iterations)',
        fontsize=12, fontweight='bold')

    for ax_idx, sp in enumerate(SPECIES):
        ax = axes[ax_idx]
        w_bio = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
        labels_sp = np.load(os.path.join(DATA_DIR, sp, 'labels.npy'))
        output_nodes_sp = np.where(labels_sp == 0)[0]
        w_in_sp = np.zeros((1, w_bio.shape[0]))
        w_in_sp[:, np.where(labels_sp == 1)[0]] = 1.0 / max(1, int(labels_sp.sum()))
        # average bio score across runs from df
        bio_mean = df[df.species == sp][f'{task_name}_bio'].mean()

        for alg in ALGORITHMS:
            hists = conv_data[sp][alg]
            if not hists: continue
            best_idx = max(range(len(hists)), key=lambda i: hists[i][-1] if hists[i] else 0)
            if task_name == 'lorenz':
                best_idx = min(range(len(hists)), key=lambda i: hists[i][-1] if hists[i] else 1)
            for i, hist in enumerate(hists):
                is_best = (i == best_idx)
                ax.plot(range(1, len(hist)+1), hist,
                        color=COLORS[alg], lw=2.0 if is_best else 0.6,
                        alpha=0.9 if is_best else 0.2,
                        label=ALG_LABELS[alg] if is_best else None)

        ax.axhline(bio_mean, color='#2196F3', lw=2, ls='--', label='Bio baseline')
        ax.set_title(sp.upper(), fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)

    # hide unused subplots
    for ax_idx in range(len(SPECIES), len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(IMAGES_DIR, f'05_convergence_{task_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: 05_convergence_{task_name}.png')

total_time = time.time() - t0_total
print(f'\nStep 03 complete.  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')
