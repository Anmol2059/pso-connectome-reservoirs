"""
Step 04 — Statistical Analysis + Graph Metrics
===============================================
Loads held-out results from steps 02 & 03 and produces:
  1. Paired t-tests + Cohen's d: PSO-A vs Bio, PSO-B vs Bio, Random vs Bio
  2. Box plots with significance markers (*, **, ***)
  3. Weighted directed graph metric comparison (clustering, path length, sigma)
  4. Input-node sensitivity analysis: 3 alternative seeds for mouse/rat/macaque

Scientific fixes applied (Reviewer 2):
  (#1)  Figure 07 uses WEIGHTED directed graph metrics. PSO only changes weights,
        so binary path length / sigma would be identical across conditions —
        using 1/weight as path distance makes path length and sigma meaningful.
  (#4)  Cohen's d (paired) reported alongside every t-test; effect size is
        informative at df=4 where statistical power is low.
  (#9)  graph_metrics uses nx.DiGraph throughout; directed mean_k formula
        n_edges/n_nodes (no factor 2); Fagiolo (2007) directed clustering.
  (#3)  Input-node sensitivity: 3 alternative seeds tested for non-fly species
        to verify baseline MC is robust to input selection.

Outputs:
  images/06_mc_boxplots.png
  images/07_graph_metrics.png
  data/stats_summary.csv       — includes Cohen's d
  data/graph_metrics.csv
  data/input_sensitivity.csv   — MC baseline across 3 input seeds
"""
import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy import stats
from sklearn.linear_model import RidgeCV
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import eigs as sparse_eigs

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── paths ─────────────────────────────────────────────────────────────────────
PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(PROJ_DIR, 'data')
IMAGES_DIR = os.path.join(PROJ_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

SPECIES = ['fly', 'mouse', 'rat', 'macaque']

COLORS = {
    'bio':    '#2196F3',
    'pso_a':  '#4CAF50',
    'pso_b':  '#FF5722',
    'random': '#9E9E9E',
}

# ── load results ──────────────────────────────────────────────────────────────
pso_df  = pd.read_csv(os.path.join(DATA_DIR, 'pso_results.csv'))
base_df = pd.read_csv(os.path.join(DATA_DIR, 'baseline_results.csv'))

# ── statistical tests ─────────────────────────────────────────────────────────
def significance_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def cohens_d_paired(a, b):
    """Cohen's d for paired samples: mean(diff)/std(diff)."""
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


stats_rows = []
print(f'\n{"Species":<10} {"Condition":<12} {"Bio μ":>8} {"Cond μ":>8} '
      f'{"Δ%":>7} {"d":>6} {"p":>10} {"sig":>5}')
print('-' * 75)

for sp in SPECIES:
    df_sp = pso_df[pso_df.species == sp]
    bio   = df_sp['mc_bio'].values

    for cond, col in [('PSO-A', 'mc_pso_a'), ('PSO-B', 'mc_pso_b'), ('Random', 'mc_random')]:
        scores    = df_sp[col].values
        t_stat, p = stats.ttest_rel(scores, bio)
        d         = cohens_d_paired(scores, bio)
        delta_pct = (scores.mean() - bio.mean()) / max(bio.mean(), 1e-6) * 100
        sig       = significance_label(p)
        stats_rows.append({
            'species': sp, 'condition': cond,
            'bio_mean': bio.mean(), 'bio_std': bio.std(),
            'cond_mean': scores.mean(), 'cond_std': scores.std(),
            'delta_pct': delta_pct, 't_stat': t_stat,
            'p_value': p, 'cohens_d': d, 'significance': sig,
            'n_runs': len(scores),
        })
        print(f'{sp:<10} {cond:<12} {bio.mean():>8.3f} {scores.mean():>8.3f} '
              f'{delta_pct:>6.1f}% {d:>6.2f} {p:>10.4f} {sig:>5}')

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(DATA_DIR, 'stats_summary.csv'), index=False)
print('\nSaved: stats_summary.csv')

# ── figure 06: box plots with significance markers ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 7), sharey=False)
fig.suptitle(
    'Memory Capacity: Biological vs PSO-Optimised vs Random\n'
    'Scores on HELD-OUT signal (PSO optimised on separate signal to prevent selection bias)\n'
    '(box = IQR, line = median, whiskers = 1.5×IQR)',
    fontsize=11, fontweight='bold')

cond_labels = ['Bio', 'PSO-A', 'PSO-B', 'Random']
cond_cols   = ['mc_bio', 'mc_pso_a', 'mc_pso_b', 'mc_random']
cond_colors = [COLORS['bio'], COLORS['pso_a'], COLORS['pso_b'], COLORS['random']]

for ax, sp in zip(axes, SPECIES):
    df_sp = pso_df[pso_df.species == sp]
    data  = [df_sp[col].values for col in cond_cols]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], cond_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    y_top  = max(np.max(d) for d in data)
    y_bot  = min(np.min(d) for d in data)
    y_step = max((y_top - y_bot) * 0.12, 0.01)
    sp_stats = stats_df[stats_df.species == sp]

    for i, cond in enumerate(['PSO-A', 'PSO-B', 'Random'], start=1):
        row = sp_stats[sp_stats.condition == cond].iloc[0]
        sig = row['significance']
        d   = row['cohens_d']
        h   = y_top + y_step * i
        ax.annotate('', xy=(i + 1, h), xytext=(1, h),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=1.0))
        label = f'{sig}' if sig != 'ns' else f'ns (d={d:.2f})'
        ax.text((1 + i + 1) / 2, h + y_step * 0.1, label,
                ha='center', va='bottom', fontsize=9, color='black')

    ax.set_title(sp.upper(), fontweight='bold', fontsize=13)
    ax.set_xticks(range(1, len(cond_labels) + 1))
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel('MC Score (Σ R², held-out)' if sp == 'fly' else '')
    ax.grid(axis='y', alpha=0.4)

patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(cond_colors, cond_labels)]
fig.legend(handles=patches, loc='lower center', ncol=4,
           fontsize=11, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '06_mc_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 06_mc_boxplots.png')


# ── graph metrics (weighted directed) ─────────────────────────────────────────
def graph_metrics(w):
    """
    Compute directed weighted graph metrics.

    - Clustering: nx.average_clustering on DiGraph with weight='weight'
      (Fagiolo 2007 directed weighted formulation).
    - Path length: weighted shortest path using 1/weight as distance
      (stronger connection → shorter effective distance). This makes path
      length differ across conditions since PSO changes weights.
    - Small-world σ: directed Watts-Strogatz formulation; mean_k = edges/nodes
      (directed formula; no factor 2 as in undirected). Fagiolo (2007).
    - Binary path length / sigma would be identical across Bio/PSO-A/PSO-B
      since PSO only changes weights, not topology (#1, #9).
    """
    n = w.shape[0]

    # Directed weighted graph for clustering (Fagiolo 2007)
    G_dir = nx.from_numpy_array(w, create_using=nx.DiGraph)
    clust = nx.average_clustering(G_dir, weight='weight')

    # Build distance graph: weight → 1/weight (higher weight = shorter distance)
    G_dist = nx.DiGraph()
    G_dist.add_nodes_from(range(n))
    for u_n, v_n, d in G_dir.edges(data=True):
        wt = d.get('weight', 0)
        if wt > 1e-10:
            G_dist.add_edge(u_n, v_n, weight=1.0 / wt)

    # Weakly connected component for path length
    if not nx.is_weakly_connected(G_dist):
        lcc = G_dist.subgraph(
            max(nx.weakly_connected_components(G_dist), key=len)).copy()
    else:
        lcc = G_dist

    try:
        avg_path = nx.average_shortest_path_length(lcc, weight='weight')
    except Exception:
        avg_path = np.nan

    # Directed small-world: mean_k = n_edges / n_nodes (no factor 2)
    n_lcc   = lcc.number_of_nodes()
    n_edges = lcc.number_of_edges()
    mean_k  = n_edges / max(n_lcc, 1)   # directed degree formula

    c_rand = max(mean_k / max(n_lcc, 1), 1e-10)
    l_rand = np.log(n_lcc) / np.log(max(mean_k, 1.1))
    gamma  = clust / c_rand
    lam    = avg_path / max(l_rand, 1e-10) if not np.isnan(avg_path) else np.nan
    sigma  = gamma / max(lam, 1e-10) if not np.isnan(lam) else np.nan

    return {'clustering': clust, 'path_length': avg_path, 'sigma': sigma}


print('\n\nGraph metrics (weighted directed, Fagiolo 2007):')
print(f'{"Species":<10} {"Condition":<10} {"Clustering":>12} {"Path(1/w)":>12} {"σ":>8}')
print('-' * 55)

metric_rows = []
for sp in SPECIES:
    w_bio = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    is_binary = bool(np.all((w_bio == 0) | (w_bio == 1)))

    pso_a_path = os.path.join(DATA_DIR, sp, 'conn_pso_a.npy')
    pso_b_path = os.path.join(DATA_DIR, sp, 'conn_pso_b.npy')

    for label, path in [('Bio', None), ('PSO-A', pso_a_path), ('PSO-B', pso_b_path)]:
        if path is not None and not os.path.exists(path):
            print(f'  {sp} {label}: connectome not found, skipping')
            continue
        w = w_bio if path is None else np.load(path)
        m = graph_metrics(w)
        m.update({'species': sp, 'condition': label, 'is_binary_source': is_binary})
        metric_rows.append(m)
        print(f'{sp:<10} {label:<10} {m["clustering"]:>12.4f} '
              f'{m["path_length"]:>12.3f}  {m["sigma"]:>8.3f}'
              + (' [binary]' if is_binary and label == 'Bio' else ''))

metric_df = pd.DataFrame(metric_rows)
metric_df.to_csv(os.path.join(DATA_DIR, 'graph_metrics.csv'), index=False)
print('\nSaved: graph_metrics.csv')

# ── figure 07: graph metrics comparison ──────────────────────────────────────
metric_names  = ['clustering', 'path_length', 'sigma']
metric_labels = [
    'Clustering Coefficient\n(weighted directed, Fagiolo 2007)',
    'Avg. Weighted Path Length\n(distance = 1/weight)',
    'Small-World Index σ\n(directed Watts-Strogatz)',
]
cond_order  = ['Bio', 'PSO-A', 'PSO-B']
bar_colors  = [COLORS['bio'], COLORS['pso_a'], COLORS['pso_b']]

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
fig.suptitle(
    'Graph Topology: Biological vs PSO-Optimised Connectomes\n'
    'All metrics use weighted directed formulation (path = 1/weight distance)\n'
    'Path length and σ differ between conditions because PSO changes weights.',
    fontsize=12, fontweight='bold')

for row, (metric, ylabel) in enumerate(zip(metric_names, metric_labels)):
    for col, sp in enumerate(SPECIES):
        ax = axes[row][col]
        is_bin = metric_df[(metric_df.species == sp)]['is_binary_source'].any()

        vals = []
        for c in cond_order:
            rows = metric_df[(metric_df.species == sp) & (metric_df.condition == c)]
            vals.append(rows[metric].values[0] if len(rows) > 0 else np.nan)

        bars = ax.bar(cond_order, vals, color=bar_colors, alpha=0.85, edgecolor='white')
        if row == 0:
            title = sp.upper()
            if is_bin:
                title += '\n(binary src)'
            ax.set_title(title, fontweight='bold', fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '07_graph_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 07_graph_metrics.png')


# ── input-node sensitivity analysis (#3) ─────────────────────────────────────
print('\n\nInput-node sensitivity analysis (3 alternative seeds for non-fly species):')
print('This verifies that MC baseline results are not specific to seed=42 input assignment.\n')

ALPHA_SEN  = 0.97
N_SEN      = 2000
WASHOUT_SEN = 200
MAX_LAG_SEN = 50
TRAIN_SEN  = 0.7
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def _sr(w):
    try:
        ev = sparse_eigs(sp_sparse.csr_matrix(w), k=1, which='LM',
                         return_eigenvectors=False, maxiter=500, tol=1e-4)
        return float(np.abs(ev[0]))
    except Exception:
        return float(np.max(np.abs(np.linalg.eigvals(w))))


def _sim(w_r, u_proj, out_nodes, washout):
    T, n = len(u_proj), w_r.shape[0]
    s = np.zeros(n)
    out = np.empty((T - washout, len(out_nodes)))
    for t in range(T):
        s = np.tanh(s @ w_r + u_proj[t])
        if t >= washout:
            out[t - washout] = s[out_nodes]
    return out


def _mc_sensitivity(w, input_nodes, run_seed=0):
    n = w.shape[0]
    output_nodes = np.array([i for i in range(n) if i not in set(input_nodes)])
    w_in = np.zeros((1, n))
    w_in[:, input_nodes] = 1.0 / max(1, len(input_nodes))
    sr = _sr(w)
    if sr < 1e-10:
        return 0.0
    w_r = (ALPHA_SEN / sr) * w
    rng = np.random.default_rng(run_seed)
    n_tot = WASHOUT_SEN + N_SEN + MAX_LAG_SEN
    u = rng.uniform(-1, 1, n_tot)
    x_full = u[MAX_LAG_SEN:].reshape(-1, 1)
    y_full = np.column_stack([
        u[MAX_LAG_SEN - k: MAX_LAG_SEN - k + WASHOUT_SEN + N_SEN]
        for k in range(1, MAX_LAG_SEN + 1)
    ])[WASHOUT_SEN:]
    u_proj = x_full @ w_in
    rs = _sim(w_r, u_proj, output_nodes, WASHOUT_SEN)
    n_tr = int(TRAIN_SEN * N_SEN)
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    model.fit(rs[:n_tr], y_full[:n_tr])
    y_pred = model.predict(rs[n_tr:])
    score = 0.0
    for k in range(MAX_LAG_SEN):
        yt, yp = y_full[n_tr:, k], y_pred[:, k]
        if np.std(yp) > 1e-10 and np.std(yt) > 1e-10:
            score += float(np.corrcoef(yt, yp)[0, 1] ** 2)
    return score


sensitivity_rows = []
INPUT_SEEDS = [42, 7, 13, 99]   # seed=42 is default; others are alternatives

for sp in SPECIES:
    if sp == 'fly':
        print(f'  {sp}: fly uses biologically motivated olfactory nodes — skipping')
        continue
    w = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    n = w.shape[0]
    n_input = max(1, int(0.2 * n))
    mc_per_seed = []
    for iseed in INPUT_SEEDS:
        rng_i = np.random.default_rng(iseed)
        inp   = rng_i.choice(n, size=n_input, replace=False)
        mc    = _mc_sensitivity(w, inp, run_seed=0)
        mc_per_seed.append(mc)
        sensitivity_rows.append({'species': sp, 'input_seed': iseed, 'mc': mc})
    cv = np.std(mc_per_seed) / (np.mean(mc_per_seed) + 1e-10) * 100
    print(f'  {sp}: MC = {np.mean(mc_per_seed):.3f} ± {np.std(mc_per_seed):.3f}  '
          f'(CV={cv:.1f}%)  seeds={INPUT_SEEDS}')

sens_df = pd.DataFrame(sensitivity_rows)
sens_df.to_csv(os.path.join(DATA_DIR, 'input_sensitivity.csv'), index=False)
print('Saved: input_sensitivity.csv')

# ── final summary ──────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('FINAL SUMMARY — PSO vs Biology (held-out evaluation)')
print('=' * 70)
print(f'{"Species":<10} {"PSO-A Δ%":>10} {"d_A":>6} {"sig_A":>6}  '
      f'{"PSO-B Δ%":>10} {"d_B":>6} {"sig_B":>6}')
print('-' * 55)
for sp in SPECIES:
    r_a = stats_df[(stats_df.species == sp) & (stats_df.condition == 'PSO-A')].iloc[0]
    r_b = stats_df[(stats_df.species == sp) & (stats_df.condition == 'PSO-B')].iloc[0]
    print(f'{sp:<10} {r_a.delta_pct:>+9.1f}% {r_a.cohens_d:>6.2f} {r_a.significance:>6}  '
          f'{r_b.delta_pct:>+9.1f}% {r_b.cohens_d:>6.2f} {r_b.significance:>6}')

print('\nNote: N=5 gives df=4; Cohen\'s d characterises effect size regardless of power.')
print('Step 04 complete.')
