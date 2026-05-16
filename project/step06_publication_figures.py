"""
Step 06 — Publication-Quality Figures
======================================
Generates visually striking, scientifically rigorous figures for the paper.

Figures produced:
  08_heatmap_improvement.png   — species × algorithm improvement-over-bio heatmap
  09_radar_charts.png          — radar (spider) charts: all 4 tasks per species
  10_connectome_graphs.png     — network topology drawings for all 7 species
  11_weight_distributions.png  — violin plots: bio vs best-algorithm weights
  12_task_correlation.png      — scatter: does good MC predict good NARMA/Lorenz?
  13_summary_dotplot.png       — forest-style dot plot: all results in one figure
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import networkx as nx
from scipy import stats

warnings.filterwarnings('ignore')

# ── publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.5,
    'lines.linewidth':   1.5,
})

PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(PROJ_DIR, 'data')
IMAGES_DIR = os.path.join(PROJ_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
ALG_COLORS  = {'bio': '#455A64', 'pso': '#2196F3', 'de': '#FF9800',
               'gwo': '#9C27B0', 'woa': '#E91E63'}
ALG_LABELS  = {'bio': 'Biological', 'pso': 'PSO', 'de': 'DE',
               'gwo': 'GWO', 'woa': 'WOA'}
TASK_COLORS = {'mc': '#1565C0', 'lorenz': '#B71C1C',
               'narma10': '#1B5E20', 'mackey_glass': '#E65100'}
TASK_LABELS = {'mc': 'Memory\nCapacity', 'lorenz': 'Lorenz\nNRMSE',
               'narma10': 'NARMA-10\nNRMSE', 'mackey_glass': 'Mackey-Glass\nNRMSE'}
TASK_LABELS_SHORT = {'mc': 'MC', 'lorenz': 'Lorenz',
                     'narma10': 'NARMA-10', 'mackey_glass': 'Mackey-Glass'}
SPECIES_LABELS = {
    'celegans':  'C. elegans',
    'fly':       'Fly',
    'mouse':     'Mouse',
    'rat':       'Rat',
    'macaque_w': 'Macaque',
    'human':     'Human',
}

TASKS      = ['mc', 'lorenz', 'narma10', 'mackey_glass']
REG_TASKS  = ['lorenz', 'narma10', 'mackey_glass']
ALGORITHMS = ['pso', 'de', 'gwo', 'woa']
SPECIES    = ['celegans', 'fly', 'mouse', 'rat', 'macaque_w', 'human']

# ── load data ─────────────────────────────────────────────────────────────────
df_path = os.path.join(DATA_DIR, 'opt_results.csv')
if not os.path.exists(df_path):
    print(f'ERROR: {df_path} not found. Run step03_pso.py first.')
    raise SystemExit(1)

df = pd.read_csv(df_path)
print(f'Loaded opt_results.csv: {len(df)} rows, {len(df.columns)} columns')
print(f'Species: {df.species.unique().tolist()}')


# ── helper: per-species per-algorithm mean ± std ──────────────────────────────
def get_stats(sp, alg, task):
    """Return (mean, std) for given species/algorithm/task on held-out evaluation."""
    col = f'{task}_bio' if alg == 'bio' else f'{task}_{alg}_a'
    if col not in df.columns:
        return np.nan, np.nan
    vals = df[df.species == sp][col].dropna().values
    if len(vals) == 0:
        return np.nan, np.nan
    return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else (vals.mean(), 0.0)


def improvement_pct(sp, alg, task):
    """% improvement over bio baseline (positive = better than bio)."""
    mu_bio, _  = get_stats(sp, 'bio', task)
    mu_alg, _  = get_stats(sp, alg, task)
    if np.isnan(mu_bio) or np.isnan(mu_alg) or abs(mu_bio) < 1e-10:
        return np.nan
    if task == 'mc':   # higher is better
        return (mu_alg - mu_bio) / mu_bio * 100
    else:              # lower NRMSE is better → improvement = reduction
        return (mu_bio - mu_alg) / mu_bio * 100


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 08 — Heatmap: improvement over bio baseline
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 08: Improvement heatmap...')

n_sp  = len(SPECIES)
n_alg = len(ALGORITHMS)
n_tsk = len(TASKS)

fig, axes = plt.subplots(1, n_tsk, figsize=(14, 5))
fig.suptitle(
    'Percentage Improvement over Biological Baseline\n'
    '(positive = better than unoptimised connectome; '
    'held-out evaluation signal)',
    fontweight='bold', fontsize=11, y=1.02)

for ax, task in zip(axes, TASKS):
    matrix = np.zeros((n_sp, n_alg))
    for i, sp in enumerate(SPECIES):
        for j, alg in enumerate(ALGORITHMS):
            matrix[i, j] = improvement_pct(sp, alg, task)

    # diverging colormap centred at 0
    vmax = np.nanpercentile(np.abs(matrix[~np.isnan(matrix)]), 95)
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=-vmax, vmax=vmax)

    # annotate cells
    for i in range(n_sp):
        for j in range(n_alg):
            val = matrix[i, j]
            if not np.isnan(val):
                txt_color = 'black' if abs(val) < vmax * 0.6 else 'white'
                ax.text(j, i, f'{val:+.0f}%', ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color=txt_color)

    ax.set_xticks(range(n_alg))
    ax.set_xticklabels([ALG_LABELS[a] for a in ALGORITHMS], fontsize=8)
    ax.set_yticks(range(n_sp))
    ax.set_yticklabels([SPECIES_LABELS[s].replace('\n', ' ') for s in SPECIES], fontsize=8)
    ax.set_title(TASK_LABELS_SHORT[task], fontweight='bold', pad=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='% improvement over bio')
    ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '08_heatmap_improvement.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('  Saved: 08_heatmap_improvement.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 09 — Radar charts: one per species, 4 tasks × 4 algorithms + bio
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 09: Radar charts...')

# Normalise to [0, 1] across all species/algorithms per task so radar is comparable
# MC: higher is better → normalise directly
# NRMSE: lower is better → normalise as 1 - normalised_nrmse

norm_min = {}
norm_max = {}
for task in TASKS:
    all_vals = []
    for sp in SPECIES:
        for alg in ['bio'] + ALGORITHMS:
            m, _ = get_stats(sp, alg, task)
            if not np.isnan(m):
                all_vals.append(m)
    norm_min[task] = min(all_vals) if all_vals else 0
    norm_max[task] = max(all_vals) if all_vals else 1


def normalise(val, task):
    lo, hi = norm_min[task], norm_max[task]
    if hi - lo < 1e-10:
        return 0.5
    n = (val - lo) / (hi - lo)
    return n if task == 'mc' else 1 - n   # invert NRMSE so higher = better


n_tasks = len(TASKS)
angles  = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
angles += angles[:1]   # close polygon

task_tick_labels = [TASK_LABELS_SHORT[t] for t in TASKS]

ncols = 4
nrows = 2
fig, axes_flat = plt.subplots(nrows, ncols, figsize=(14, 7),
                               subplot_kw=dict(polar=True))
fig.suptitle(
    'Multi-Task Performance Radar — All Species\n'
    '(radial axis normalised per task; MC: higher = better; NRMSE: higher = less error)',
    fontweight='bold', fontsize=11, y=1.03)

axes_all = axes_flat.flatten()

for ax_idx, sp in enumerate(SPECIES):
    ax = axes_all[ax_idx]

    for alg in ['bio'] + ALGORITHMS:
        vals = []
        for task in TASKS:
            m, _ = get_stats(sp, alg, task)
            vals.append(normalise(m, task) if not np.isnan(m) else 0.0)
        vals += vals[:1]   # close

        lw  = 2.2 if alg == 'bio' else 1.4
        ls  = '--' if alg == 'bio' else '-'
        col = ALG_COLORS[alg]
        ax.plot(angles, vals, color=col, lw=lw, ls=ls,
                label=ALG_LABELS[alg], zorder=3)
        ax.fill(angles, vals, color=col, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_tick_labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_title(SPECIES_LABELS[sp].replace('\n', ' '), fontweight='bold',
                 fontsize=9, pad=12)
    ax.grid(color='grey', alpha=0.3, linewidth=0.5)
    ax.spines['polar'].set_visible(False)

# Hide unused subplot
if len(SPECIES) < len(axes_all):
    for ax in axes_all[len(SPECIES):]:
        ax.set_visible(False)

# Shared legend
handles = [Line2D([0], [0], color=ALG_COLORS[a], lw=2,
                  ls='--' if a == 'bio' else '-',
                  label=ALG_LABELS[a])
           for a in ['bio'] + ALGORITHMS]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.04),
           frameon=True, edgecolor='grey')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '09_radar_charts.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('  Saved: 09_radar_charts.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Connectome network visualisations
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 10: Connectome network graphs...')

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    'Biological Connectome Architectures\n'
    '(node colour: input=teal, output=coral, internal=grey; '
    'edge width ∝ log(1+weight); top 5 % strongest edges shown)',
    fontweight='bold', fontsize=11, y=1.02)

axes_flat = axes.flatten()

for ax_idx, sp in enumerate(SPECIES):
    ax = axes_flat[ax_idx]
    ax.set_facecolor('#FAFAFA')

    w   = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    lbl = np.load(os.path.join(DATA_DIR, sp, 'labels.npy'))
    n   = w.shape[0]

    input_nodes  = set(np.where(lbl == 1)[0].tolist())
    output_nodes = set(np.where(lbl == 0)[0].tolist())

    # Build directed graph
    G = nx.from_numpy_array(w, create_using=nx.DiGraph)

    # Show only top-5% strongest edges to avoid visual clutter
    all_weights = w[w > 0].flatten()
    thresh = np.percentile(all_weights, 95) if len(all_weights) > 0 else 0
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True)
                     if d['weight'] >= thresh]
    edge_weights  = [G[u][v]['weight'] for u, v in edges_to_draw]
    w_max = max(edge_weights) if edge_weights else 1.0
    edge_widths   = [0.4 + 2.5 * np.log1p(ew) / np.log1p(w_max)
                     for ew in edge_weights]

    # Node colours
    node_colors = []
    for i in range(n):
        if i in input_nodes:
            node_colors.append('#00897B')   # teal — input
        elif i in output_nodes:
            node_colors.append('#EF5350')   # coral — output
        else:
            node_colors.append('#90A4AE')   # grey — internal

    # Layout: circular for small (<= 80 nodes), spring for larger
    if n <= 80:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n))

    node_size = max(20, min(200, 2000 // n))

    nx.draw_networkx_edges(
        G, pos, edgelist=edges_to_draw, ax=ax,
        width=edge_widths, alpha=0.45,
        edge_color='#546E7A', arrows=False)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_size,
        linewidths=0.3, edgecolors='white')
    ax.set_title(
        f'{SPECIES_LABELS[sp].replace(chr(10), " ")}\n'
        f'{n} nodes, {int((w > 0).sum())} edges',
        fontweight='bold', fontsize=9)
    ax.axis('off')
    ax.set_aspect('equal')

# Hide the empty slot (6 species in 2×4 grid leaves slot 6 empty)
axes_flat[6].set_visible(False)

# Legend for node colours
legend_patches = [
    mpatches.Patch(color='#00897B', label='Input nodes'),
    mpatches.Patch(color='#EF5350', label='Output nodes'),
    mpatches.Patch(color='#90A4AE', label='Internal nodes'),
]
axes_flat[-1].set_visible(True)
axes_flat[-1].axis('off')
axes_flat[-1].legend(handles=legend_patches, loc='center',
                     fontsize=10, frameon=True, edgecolor='grey',
                     title='Node type', title_fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '10_connectome_graphs.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('  Saved: 10_connectome_graphs.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Weight distributions: bio vs best algorithm per species
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 11: Weight distributions...')

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle(
    'Edge Weight Distributions: Biological vs Bio-Inspired Optimised\n'
    '(violin = kernel density; white dot = median; box = IQR)',
    fontweight='bold', fontsize=11, y=1.02)

axes_flat = axes.flatten()

for ax_idx, sp in enumerate(SPECIES):
    ax = axes_flat[ax_idx]
    w_bio = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    nz_bio = w_bio[w_bio > 0].flatten()
    is_bin = bool(np.all((w_bio == 0) | (w_bio == 1)))

    datasets   = [nz_bio]
    tick_lbls  = ['Bio']
    tick_cols  = [ALG_COLORS['bio']]

    for alg in ALGORITHMS:
        path = os.path.join(DATA_DIR, sp, f'conn_{alg}_a.npy')
        if os.path.exists(path):
            w_opt = np.load(path)
            nz_opt = w_opt[w_opt > 0].flatten()
            if len(nz_opt) > 0:
                datasets.append(nz_opt)
                tick_lbls.append(ALG_LABELS[alg])
                tick_cols.append(ALG_COLORS[alg])

    if len(datasets) < 2:
        ax.text(0.5, 0.5, 'No optimised\nweights found',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(SPECIES_LABELS[sp].replace('\n', ' '), fontweight='bold')
        continue

    # log-transform for heavily skewed distributions (fly has weights up to 4477)
    plot_data = [np.log1p(d) for d in datasets]

    parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                          showmedians=True, showextrema=False,
                          widths=0.7)

    for i, (pc, col) in enumerate(zip(parts['bodies'], tick_cols)):
        pc.set_facecolor(col)
        pc.set_edgecolor('white')
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(1.8)

    ax.set_xticks(range(len(tick_lbls)))
    ax.set_xticklabels(tick_lbls, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('log(1 + weight)', fontsize=8)
    ax.set_title(
        f'{SPECIES_LABELS[sp].replace(chr(10), " ")}'
        + (' [binary src]' if is_bin else ''),
        fontweight='bold', fontsize=9)

axes_flat[-1].set_visible(True)
axes_flat[-1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '11_weight_distributions.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('  Saved: 11_weight_distributions.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Cross-task correlation: does MC predict NRMSE performance?
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 12: Cross-task correlation...')

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle(
    'Cross-Task Correlation: Memory Capacity vs NRMSE Tasks\n'
    '(each point = one species × algorithm × run; r = Pearson; '
    'shaded = 95% CI)',
    fontweight='bold', fontsize=11)

for ax, reg_task in zip(axes, REG_TASKS):
    xs, ys, colors, species_per_point = [], [], [], []

    for sp in SPECIES:
        for alg in ['bio'] + ALGORITHMS:
            mc_col  = 'mc_bio'  if alg == 'bio' else f'mc_{alg}_a'
            nrm_col = f'{reg_task}_bio' if alg == 'bio' else f'{reg_task}_{alg}_a'
            if mc_col not in df.columns or nrm_col not in df.columns:
                continue
            sub = df[df.species == sp][[mc_col, nrm_col]].dropna()
            for _, row in sub.iterrows():
                xs.append(row[mc_col])
                ys.append(row[nrm_col])
                colors.append(ALG_COLORS[alg])

    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 3:
        ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        continue

    ax.scatter(xs, ys, c=colors, alpha=0.55, s=18, linewidths=0, zorder=3)

    # Regression line + CI
    r, p = stats.pearsonr(xs, ys)
    m, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(xs.min(), xs.max(), 200)
    y_line = m * x_line + b

    # Bootstrap CI
    n_boot = 500
    boot_lines = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(len(xs), size=len(xs))
        mb, bb = np.polyfit(xs[idx], ys[idx], 1)
        boot_lines.append(mb * x_line + bb)
    ci_lo = np.percentile(boot_lines, 2.5, axis=0)
    ci_hi = np.percentile(boot_lines, 97.5, axis=0)

    ax.plot(x_line, y_line, color='#333333', lw=1.8, zorder=4)
    ax.fill_between(x_line, ci_lo, ci_hi, color='#333333', alpha=0.12, zorder=2)

    p_str = f'p<0.001' if p < 0.001 else f'p={p:.3f}'
    ax.text(0.04, 0.96, f'r = {r:.2f}, {p_str}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.8))

    ax.set_xlabel('Memory Capacity (MC)', fontsize=9)
    ax.set_ylabel(f'NRMSE — {TASK_LABELS_SHORT[reg_task]}', fontsize=9)
    ax.set_title(f'MC vs {TASK_LABELS_SHORT[reg_task]}', fontweight='bold')

# Shared legend
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=ALG_COLORS[a], markersize=7,
                  label=ALG_LABELS[a])
           for a in ['bio'] + ALGORITHMS]
fig.legend(handles=handles, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, '12_task_correlation.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('  Saved: 12_task_correlation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Summary dot plot (forest-style): all results at a glance
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 13: Summary dot plot...')

n_sp  = len(SPECIES)
n_tsk = len(TASKS)
n_alg = len(['bio'] + ALGORITHMS)   # 5 conditions

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(1, n_tsk, wspace=0.45)

for t_idx, task in enumerate(TASKS):
    ax = fig.add_subplot(gs[t_idx])

    # y positions: species spaced by 6, algorithms offset within each species
    offsets = np.linspace(-1.0, 1.0, n_alg)
    yticks, yticklabels = [], []

    for s_idx, sp in enumerate(SPECIES):
        y_base = s_idx * 6
        yticks.append(y_base)
        yticklabels.append(SPECIES_LABELS[sp].replace('\n', ' '))

        ax.axhline(y_base, color='#ECEFF1', lw=8, zorder=0)   # zebra stripe

        for a_idx, alg in enumerate(['bio'] + ALGORITHMS):
            col = f'{task}_bio' if alg == 'bio' else f'{task}_{alg}_a'
            if col not in df.columns:
                continue
            vals = df[df.species == sp][col].dropna().values
            if len(vals) == 0:
                continue

            mu  = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            y   = y_base + offsets[a_idx]

            ax.errorbar(mu, y, xerr=1.96 * sem,
                        fmt='o', color=ALG_COLORS[alg],
                        markersize=5.5, capsize=3, capthick=1.2,
                        linewidth=1.2, zorder=5)

    ax.set_yticks(yticks)
    if t_idx == 0:
        ax.set_yticklabels(yticklabels, fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-3, n_sp * 6 - 2)
    ax.set_xlabel(
        'MC Score' if task == 'mc' else 'NRMSE', fontsize=9)
    ax.set_title(TASK_LABELS_SHORT[task], fontweight='bold', fontsize=10)
    ax.invert_yaxis()

    # For NRMSE tasks, lower is better — add annotation
    if task in REG_TASKS:
        ax.text(0.98, 0.01, '← better', transform=ax.transAxes,
                fontsize=7, ha='right', va='bottom', color='grey',
                style='italic')
    else:
        ax.text(0.98, 0.01, 'better →', transform=ax.transAxes,
                fontsize=7, ha='right', va='bottom', color='grey',
                style='italic')

# Shared legend
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=ALG_COLORS[a], markersize=7,
                  label=ALG_LABELS[a])
           for a in ['bio'] + ALGORITHMS]
fig.legend(handles=handles, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=True,
           title='Algorithm (Bio-A initialisation)', title_fontsize=9)
fig.suptitle(
    'All Species × All Algorithms × All Tasks — Mean ± 95% CI (10 runs)\n'
    'Dots = mean held-out score; bars = 1.96 × SEM',
    fontweight='bold', fontsize=11, y=1.02)

plt.savefig(os.path.join(IMAGES_DIR, '13_summary_dotplot.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('  Saved: 13_summary_dotplot.png')

print('\nAll publication figures saved to:', IMAGES_DIR)
print('Run after step03_pso.py completes.')
