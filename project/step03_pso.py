"""
Step 03 — Multi-Algorithm Bio-Inspired Optimisation of Connectome Reservoirs
=============================================================================
Optimises biological connectome edge weights with four bio-inspired algorithms:

  PSO  — Particle Swarm Optimisation   (Kennedy & Eberhart 1995)
           Inspired by collective movement of birds/fish schools.
  DE   — Differential Evolution        (Storn & Price 1997)
           Inspired by Darwinian genetic recombination and mutation.
  GWO  — Grey Wolf Optimiser           (Mirjalili et al. 2014)
           Inspired by grey wolf pack hunting hierarchy (alpha/beta/delta).
  WOA  — Whale Optimisation Algorithm  (Mirjalili & Lewis 2016)
           Inspired by humpback whale bubble-net hunting strategy.

Tasks:
  MC           — Memory Capacity (Σ R² over 50 lags; higher = better)
  Lorenz       — Lorenz attractor NRMSE one-step-ahead (lower = better)
  NARMA-10     — Nonlinear autoregressive moving-average NRMSE (lower = better)
  Mackey-Glass — Mackey-Glass delay-differential chaotic NRMSE (lower = better)

Conditions per species:
  Bio-A   — Optimiser initialised near biological weights (Gaussian perturbation)
  Rand-B  — PSO only, random init (MC task only; negative control)
  Random  — Random re-weighting, no optimisation (null baseline)
  Bio     — Unoptimised biological weights (held-out evaluation)

Scientific design:
  Held-out evaluation: optimisers search on signal A (seed=run),
  all reported scores use signal B (seed=run+EVAL_SEED_OFFSET).
  All four algorithms use GPU-batched objectives for fair comparison.
  Identical population: N_PARTICLES=20, N_ITER=50 per run.

Outputs:
  data/opt_results.csv              — all conditions × algorithms × tasks × runs
  data/{sp}/conn_{alg}_a.npy        — best MC weights per species/algorithm
  images/05_convergence_{task}.png  — convergence curves per task
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
SPECIES = ['celegans', 'fly', 'mouse', 'rat', 'macaque_w', 'human']

N_RUNS       = 10
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
DE_F         = 0.8
DE_CR        = 0.9
WEIGHT_SCALE = 3.0
RIDGE_ALPHA  = 1.0

TASKS            = ['mc', 'lorenz', 'narma10', 'mackey_glass']
REGRESSION_TASKS = ['lorenz', 'narma10', 'mackey_glass']  # NRMSE, lower = better
ALGORITHMS       = ['pso', 'de', 'gwo', 'woa']

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
    x_full = traj[:, 0:1]
    x_in   = x_full[:-1]            # (WASHOUT+N_SAMPLES, 1)
    y_out  = x_full[1:, 0]          # (WASHOUT+N_SAMPLES,)
    mu, std = x_in.mean(), x_in.std() + 1e-10
    x_in  = (x_in  - mu) / std
    y_out = (y_out - mu) / std
    return x_in, y_out


# ── NARMA-10 ───────────────────────────────────────────────────────────────────
def make_narma10_data(seed):
    """
    Nonlinear AutoRegressive Moving Average order-10 system.
    y(t) = 0.3*y(t-1) + 0.05*y(t-1)*Σ_{i=1}^{10}y(t-i) + 1.5*u(t-10)*u(t-1) + 0.1
    Input: u ~ Uniform[0, 0.5].  Target: y(t) (system identification, not prediction).
    Atiya & Parlos 2000; Jaeger 2001.
    """
    n_total = WASHOUT + N_SAMPLES
    n_buf   = 30   # burn-in buffer so initial y=0 doesn't corrupt targets
    n_gen   = n_buf + n_total
    rng     = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_gen)
    y = np.zeros(n_gen)
    for t in range(10, n_gen):
        y[t] = (0.3 * y[t-1]
                + 0.05 * y[t-1] * float(np.sum(y[t-10:t]))
                + 1.5 * u[t-10] * u[t-1]
                + 0.1)
    x_in  = u[n_buf:].reshape(-1, 1)   # (WASHOUT+N_SAMPLES, 1)
    y_out = y[n_buf:]                    # (WASHOUT+N_SAMPLES,)
    mu_u, std_u = x_in.mean(), x_in.std() + 1e-10
    mu_y, std_y = y_out.mean(), y_out.std() + 1e-10
    return (x_in - mu_u) / std_u, (y_out - mu_y) / std_y


# ── Mackey-Glass ───────────────────────────────────────────────────────────────
def make_mackey_glass_data(seed, tau=17, dt=0.1):
    """
    Mackey-Glass delay-differential equation (tau=17, chaotic regime).
    dx/dt = 0.2*x(t-tau)/(1+x(t-tau)^10) - 0.1*x(t)
    One-step-ahead prediction task.  Mackey & Glass 1977; Glass & Mackey 1988.
    """
    tau_steps = int(round(tau / dt))  # 170 delay steps
    n_total   = WASHOUT + N_SAMPLES + 1
    n_warmup  = 500   # discard transient
    n_gen     = n_warmup + n_total

    rng = np.random.default_rng(seed)
    x0  = 1.2 + 0.1 * rng.standard_normal()
    x   = np.zeros(n_gen + tau_steps)
    x[:tau_steps] = x0

    for t in range(tau_steps, n_gen + tau_steps):
        x_del  = x[t - tau_steps]
        x[t]   = x[t-1] + dt * (0.2 * x_del / (1.0 + x_del**10) - 0.1 * x[t-1])

    series = x[tau_steps + n_warmup:]   # (n_total,)
    x_in   = series[:-1].reshape(-1, 1) # (WASHOUT+N_SAMPLES, 1)
    y_out  = series[1:]                  # (WASHOUT+N_SAMPLES,)
    mu, std = x_in.mean(), x_in.std() + 1e-10
    return (x_in - mu) / std, (y_out - mu) / std


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


TASK_DATA_FN = {
    'mc':           make_mc_data,
    'lorenz':       make_lorenz_data,
    'narma10':      make_narma10_data,
    'mackey_glass': make_mackey_glass_data,
}

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


# ── Ridge readouts ─────────────────────────────────────────────────────────────
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


def ridge_regression_nrmse(rs, y_tgt, n_train):
    """Shared NRMSE readout for Lorenz, NARMA-10, Mackey-Glass."""
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


def score_regression(w_raw, w_in, output_nodes, u_proj, y_tgt):
    """Shared NRMSE scorer for Lorenz, NARMA-10, Mackey-Glass.
    y_tgt must have shape (WASHOUT+N_SAMPLES,); washout is trimmed internally."""
    try:
        sr = spectral_radius(w_raw)
        if sr < 1e-10: return 1.0
        w_r     = (ALPHA / sr) * w_raw
        rs      = sim_esn(w_r, u_proj, output_nodes, WASHOUT)
        n_train = int(TRAIN_FRAC * N_SAMPLES)
        return ridge_regression_nrmse(rs, y_tgt[WASHOUT:], n_train)
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
    Wv    = torch.bmm(W_batch, v)
    sigma = (v * Wv).sum(dim=1).squeeze(-1).abs()
    return sigma


def _batch_sim_esn(W_scaled, u_t, out_idx, washout):
    P, n, _ = W_scaled.shape
    T       = u_t.shape[0]
    T_use   = T - washout
    s       = torch.zeros(P, n, device=W_scaled.device, dtype=W_scaled.dtype)
    out     = torch.zeros(P, T_use, len(out_idx), device=W_scaled.device, dtype=W_scaled.dtype)
    for t in range(T):
        u_step = u_t[t].unsqueeze(0).expand(P, -1)
        s = torch.tanh(torch.bmm(s.unsqueeze(1), W_scaled).squeeze(1) + u_step)
        if t >= washout:
            out[:, t - washout, :] = s[:, out_idx]
    return out.cpu().numpy()


def make_batch_objective_mc(n, nz_rows, nz_cols, w_in, output_nodes, u_proj, y_tgt):
    """Returns -MC cost (minimise -MC = maximise MC)."""
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


def make_batch_objective_regression(n, nz_rows, nz_cols, w_in, output_nodes, u_proj, y_tgt):
    """Returns NRMSE cost (minimise directly). Shared for Lorenz/NARMA-10/Mackey-Glass."""
    y_tgt_trimmed = y_tgt[WASHOUT:]

    if not USE_GPU:
        def cpu_obj(particles):
            return np.array([
                score_regression(
                    weights_from_particle(p, n, nz_rows, nz_cols),
                    w_in, output_nodes, u_proj, y_tgt)
                for p in particles])
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
        return np.array([ridge_regression_nrmse(rs_np[i], y_tgt_trimmed, n_train) for i in range(P)])

    return gpu_obj


def weights_from_particle(p, n, nz_rows, nz_cols):
    w = np.zeros((n, n))
    w[nz_rows, nz_cols] = np.abs(p)
    return w


# ── PSO ───────────────────────────────────────────────────────────────────────
def run_pso(objective, bounds_lb, bounds_ub, n_particles, n_iter, init_pos, label):
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

        trial_scores = objective(trials)
        improved     = trial_scores < scores
        pop[improved]    = trials[improved]
        scores[improved] = trial_scores[improved]

        history.append(-scores.min())
        pbar.set_postfix(best=f'{history[-1]:.3f}')
        pbar.update(1)
    pbar.close()

    best_idx = scores.argmin()
    return pop[best_idx], -scores[best_idx], history


# ── GWO (Mirjalili et al. 2014) ──────────────────────────────────────────────
def run_gwo(objective, bounds_lb, bounds_ub, n_wolves, n_iter, init_pos, label, seed=0):
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
        a = 2.0 * (1.0 - t / n_iter)

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
        scores = objective(wolves)

        si = np.argsort(scores)
        if scores[si[0]] < alpha_score:
            alpha_score = scores[si[0]]
            alpha_pos   = wolves[si[0]].copy()
            beta_pos    = wolves[si[1]].copy()
            delta_pos   = wolves[si[2]].copy()

        history.append(-alpha_score)
        pbar.set_postfix(best=f'{history[-1]:.3f}')
        pbar.update(1)
    pbar.close()

    return alpha_pos, -alpha_score, history


# ── WOA (Mirjalili & Lewis 2016) ──────────────────────────────────────────────
def run_woa(objective, bounds_lb, bounds_ub, n_whales, n_iter, init_pos, label, seed=0):
    """
    Vectorised Whale Optimisation Algorithm.
    Encircling prey, bubble-net attack (spiral), and random search in one batch.
    Mirjalili S & Lewis A (2016) Adv Eng Softw 95:51-67.
    """
    n_dims = len(bounds_lb)
    rng    = np.random.default_rng(seed)
    whales = np.clip(init_pos.copy(), bounds_lb, bounds_ub)
    scores = objective(whales)

    best_idx   = int(np.argmin(scores))
    best_pos   = whales[best_idx].copy()
    best_score = scores[best_idx]

    history = []
    pbar = tqdm(total=n_iter, desc=label, unit='iter', leave=False)

    for t in range(n_iter):
        a   = 2.0 * (1.0 - t / n_iter)   # linearly 2 → 0
        b   = 1.0                          # spiral shape constant

        r1  = rng.random((n_whales, n_dims))
        r2  = rng.random((n_whales, n_dims))
        A   = 2.0 * a * r1 - a            # (N, D)
        C   = 2.0 * r2                     # (N, D)
        p   = rng.random(n_whales)         # (N,)   switch encircle ↔ spiral
        l   = rng.uniform(-1, 1, (n_whales, n_dims))

        # Encircling prey
        D_enc = np.abs(C * best_pos - whales)
        enc   = best_pos - A * D_enc

        # Random search (exploration, |A| >= 1)
        rand_idx = rng.integers(n_whales, size=n_whales)
        rand_pos = whales[rand_idx]
        D_rand   = np.abs(C * rand_pos - whales)
        rnd      = rand_pos - A * D_rand

        # Bubble-net spiral attack
        D_spi = np.abs(best_pos - whales)
        spi   = D_spi * np.exp(b * l) * np.cos(2.0 * np.pi * l) + best_pos

        A_norm = np.abs(A).mean(axis=1)    # (N,)
        # Rule: p < 0.5 → encircle/random, p >= 0.5 → spiral
        use_spiral  = (p >= 0.5)[:, None]
        use_random  = (~use_spiral.squeeze(-1) & (A_norm >= 1))[:, None]
        use_encircle = (~use_spiral.squeeze(-1) & (A_norm < 1))[:, None]

        new_whales = (use_encircle * enc
                      + use_random  * rnd
                      + use_spiral  * spi)
        whales = np.clip(new_whales, bounds_lb, bounds_ub)
        scores = objective(whales)

        idx = int(np.argmin(scores))
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_pos   = whales[idx].copy()

        history.append(-best_score)
        pbar.set_postfix(best=f'{history[-1]:.3f}')
        pbar.update(1)

    pbar.close()
    return best_pos, -best_score, history


OPTIMIZER_FN = {'pso': run_pso, 'de': run_de, 'gwo': run_gwo, 'woa': run_woa}


# ── main loop ─────────────────────────────────────────────────────────────────
all_rows = []
conv     = {task: {sp: {alg: [] for alg in ALGORITHMS} for sp in SPECIES}
            for task in TASKS}
t0_total = time.time()

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

    input_nodes  = np.where(labels == 1)[0]
    output_nodes = np.where(labels == 0)[0]

    w_in = np.zeros((1, n))
    w_in[:, input_nodes] = 1.0 / max(1, len(input_nodes))

    print(f'\n{"="*65}')
    print(f'SPECIES: {sp.upper()}  | {n} nodes | {n_weights} edges | '
          f'{"BINARY" if is_binary else f"max_w={max_w:.3f}"}')

    for run in range(N_RUNS):
        print(f'\n  Run {run+1}/{N_RUNS}')
        init_rng = np.random.default_rng(run * 997 + 31)

        # ── generate data for all tasks (opt + held-out eval) ──────────────
        _data_opt  = {t: TASK_DATA_FN[t](seed=run)                    for t in TASKS}
        _data_eval = {t: TASK_DATA_FN[t](seed=run + EVAL_SEED_OFFSET) for t in TASKS}
        x_opt  = {t: _data_opt[t][0]  for t in TASKS}
        y_opt  = {t: _data_opt[t][1]  for t in TASKS}
        x_eval = {t: _data_eval[t][0] for t in TASKS}
        y_eval = {t: _data_eval[t][1] for t in TASKS}

        # project inputs onto reservoir input nodes
        u_opt  = {t: x_opt[t]  @ w_in for t in TASKS}
        u_eval = {t: x_eval[t] @ w_in for t in TASKS}

        # ── Bio-A / Rand-B initialisations ─────────────────────────────────
        noise  = init_rng.normal(0, MAX_PERTURB * bio_weights.std(),
                                 size=(N_PARTICLES, n_weights))
        init_a = np.clip(bio_weights[np.newaxis, :] + noise, 0, max_w)
        init_b = init_rng.uniform(0, max_w, size=(N_PARTICLES, n_weights))

        # ── batch objectives for optimisation ──────────────────────────────
        obj_opt = {}
        obj_opt['mc'] = make_batch_objective_mc(
            n, nz_rows, nz_cols, w_in, output_nodes, u_opt['mc'], y_opt['mc'])
        for task in REGRESSION_TASKS:
            obj_opt[task] = make_batch_objective_regression(
                n, nz_rows, nz_cols, w_in, output_nodes, u_opt[task], y_opt[task])

        row = {'species': sp, 'run': run,
               'n_nodes': n, 'n_edges': n_weights, 'is_binary': is_binary}

        # ── Bio (unoptimised) on held-out signals ──────────────────────────
        w_bio_eval = w_bio.copy()
        row['mc_bio'] = score_mc(w_bio_eval, w_in, output_nodes,
                                 u_eval['mc'], y_eval['mc'])
        for task in REGRESSION_TASKS:
            row[f'{task}_bio'] = score_regression(
                w_bio_eval, w_in, output_nodes, u_eval[task], y_eval[task])

        # ── Random null ────────────────────────────────────────────────────
        w_rand = np.zeros((n, n))
        w_rand[nz_rows, nz_cols] = init_rng.uniform(0, max_w, n_weights)
        row['mc_random'] = score_mc(w_rand, w_in, output_nodes,
                                    u_eval['mc'], y_eval['mc'])
        for task in REGRESSION_TASKS:
            row[f'{task}_random'] = score_regression(
                w_rand, w_in, output_nodes, u_eval[task], y_eval[task])

        # ── PSO-B (random init, MC only — negative control) ────────────────
        best_b, _, _ = run_pso(
            obj_opt['mc'], bounds_lb, bounds_ub, N_PARTICLES, N_ITER,
            init_b.copy(), f'{sp.upper()} PSO-B MC r{run}')
        w_best_b = weights_from_particle(best_b, n, nz_rows, nz_cols)
        row['mc_pso_b'] = score_mc(w_best_b, w_in, output_nodes,
                                   u_eval['mc'], y_eval['mc'])

        # ── Bio-A: all algorithms × all tasks ─────────────────────────────
        for alg in ALGORITHMS:
            fn        = OPTIMIZER_FN[alg]
            alg_seed  = run * 997 + 31 + hash(alg) % 100
            needs_seed = (alg != 'pso')

            for task_idx, task in enumerate(TASKS):
                lbl    = f'{sp.upper()} {alg.upper()}-A {task.upper()} r{run}'
                kwargs = {'seed': alg_seed + task_idx} if needs_seed else {}
                best_pos, _, hist = fn(
                    obj_opt[task], bounds_lb, bounds_ub, N_PARTICLES, N_ITER,
                    init_a.copy(), lbl, **kwargs)

                w_best = weights_from_particle(best_pos, n, nz_rows, nz_cols)
                if task == 'mc':
                    row[f'mc_{alg}_a'] = score_mc(
                        w_best, w_in, output_nodes, u_eval['mc'], y_eval['mc'])
                else:
                    row[f'{task}_{alg}_a'] = score_regression(
                        w_best, w_in, output_nodes, u_eval[task], y_eval[task])

                conv[task][sp][alg].append(hist)

                if run == 0 and task == 'mc':
                    sp_dir = os.path.join(DATA_DIR, sp)
                    np.save(os.path.join(sp_dir, f'conn_{alg}_a.npy'), w_best)

        all_rows.append(row)

        print(f'    MC:    bio={row["mc_bio"]:.3f}  '
              + '  '.join(f'{a}={row.get(f"mc_{a}_a", 0):.3f}' for a in ALGORITHMS))
        for task in REGRESSION_TASKS:
            print(f'    {task.upper()[:10]:10s}: bio={row[f"{task}_bio"]:.4f}  '
                  + '  '.join(f'{a}={row.get(f"{task}_{a}_a", 1):.4f}' for a in ALGORITHMS))

    elapsed = time.time() - t0_sp
    print(f'\n  {sp.upper()} done ({elapsed:.0f}s = {elapsed/60:.1f} min)')

# ── save results ───────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(DATA_DIR, 'opt_results.csv'), index=False)
print(f'\nSaved: opt_results.csv  ({len(df)} rows)')

# ── convergence plots ──────────────────────────────────────────────────────────
COLORS     = {'pso': '#4CAF50', 'de': '#FF9800', 'gwo': '#9C27B0', 'woa': '#E91E63'}
ALG_LABELS = {'pso': 'PSO', 'de': 'DE', 'gwo': 'GWO', 'woa': 'WOA'}
TASK_YLABS = {
    'mc':           'MC Score (higher = better)',
    'lorenz':       'NRMSE — Lorenz (lower = better)',
    'narma10':      'NRMSE — NARMA-10 (lower = better)',
    'mackey_glass': 'NRMSE — Mackey-Glass (lower = better)',
}

for task_name in TASKS:
    is_nrmse = (task_name in REGRESSION_TASKS)
    ylab     = TASK_YLABS[task_name]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    fig.suptitle(
        f'Bio-Inspired Optimiser Convergence — {task_name.upper()} Task\n'
        f'All {N_RUNS} runs (light) + best run highlighted  '
        f'({N_PARTICLES} particles × {N_ITER} iterations)',
        fontsize=12, fontweight='bold')

    for ax_idx, sp in enumerate(SPECIES):
        ax = axes[ax_idx]
        bio_mean = df[df.species == sp][f'{task_name}_bio'].mean()

        for alg in ALGORITHMS:
            hists = conv[task_name][sp][alg]
            if not hists: continue
            # best run = one with highest final history value (works for both MC and NRMSE
            # because history = -cost, so higher = better for all tasks)
            best_idx = max(range(len(hists)),
                           key=lambda i: hists[i][-1] if hists[i] else -np.inf)

            for i, hist in enumerate(hists):
                is_best  = (i == best_idx)
                # for NRMSE tasks, history = -NRMSE; negate for display
                plot_val = [-v for v in hist] if is_nrmse else hist
                ax.plot(range(1, len(plot_val)+1), plot_val,
                        color=COLORS[alg], lw=2.0 if is_best else 0.6,
                        alpha=0.9 if is_best else 0.2,
                        label=ALG_LABELS[alg] if is_best else None)

        ax.axhline(bio_mean, color='#2196F3', lw=2, ls='--', label='Bio baseline')
        ax.set_title(sp.upper(), fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)

    for ax_idx in range(len(SPECIES), len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(IMAGES_DIR, f'05_convergence_{task_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: 05_convergence_{task_name}.png')

total_time = time.time() - t0_total
print(f'\nStep 03 complete.  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')
