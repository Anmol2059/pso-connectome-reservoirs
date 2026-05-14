"""
Step 05 — Generate LLM-Ready Report Summary
============================================
Compiles all quantitative results, figure descriptions, methods, and
reviewer-response notes into data/summary.txt — a single text file
suitable for feeding to an LLM to write a full scientific report.

Outputs:
  data/summary.txt
"""
import os, json, textwrap
import numpy as np
import pandas as pd

PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(PROJ_DIR, 'data')
IMAGES_DIR = os.path.join(PROJ_DIR, 'images')
OUT_FILE   = os.path.join(DATA_DIR, 'summary.txt')

SPECIES = ['fly', 'mouse', 'rat', 'macaque']

# ── helpers ───────────────────────────────────────────────────────────────────
def _load(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _sec(title):
    bar = '=' * 70
    return f'\n{bar}\n  {title}\n{bar}\n'


def _sub(title):
    bar = '-' * 60
    return f'\n{bar}\n  {title}\n{bar}\n'


lines = []
lines.append('PSO-OPTIMISED CONNECTOME RESERVOIRS — FULL EXPERIMENTAL SUMMARY')
lines.append('Generated for LLM report writing. All quantitative results included.')
lines.append('=' * 70)

# ── project overview ──────────────────────────────────────────────────────────
lines.append(_sec('PROJECT OVERVIEW'))
lines.append(textwrap.dedent("""
Research Question:
  Can Particle Swarm Optimization (PSO) match or exceed the computational capacity
  of real biological connectomes on memory and chaotic prediction tasks?
  And if it does, does PSO rediscover the same weighted graph structure that
  millions of years of evolution built?

Approach:
  - Four real biological connectomes (fly, mouse, rat, macaque) used as reservoirs
    in Echo State Networks (ESNs).
  - Three PSO conditions per species:
      PSO-A: optimise weights starting from biological weights (fine-tuning)
      PSO-B: optimise weights starting from random weights (from scratch)
      Random: random re-weighting of same topology (null control)
  - Two benchmark tasks:
      Memory Capacity (MC): sum of R² for delayed signal recall up to 50 lags
      Lorenz prediction: one-step-ahead NRMSE on chaotic x-trajectory

Key design decisions:
  - PSO preserves biological sparsity pattern (only non-zero edges are optimised)
  - Washout = 200 steps; MAX_LAG = 50; spectral radius ρ = 0.97
  - Held-out evaluation: PSO optimised on signal A, all reported scores on signal B
    (prevents selection bias from evaluating PSO on its own optimisation target)
  - N = 5 independent runs per condition (gives paired t-test with df=4)
  - Cohen's d reported to characterise effect size given low statistical power
""").strip())

# ── connectome statistics ─────────────────────────────────────────────────────
lines.append(_sec('CONNECTOME STATISTICS'))
lines.append(f'{"Species":<12} {"Nodes":>6} {"Edges":>8} {"Density":>9} '
             f'{"Symmetric":>10} {"Binary":>8} {"w_min":>8} {"w_max":>8} {"Inputs":>8}')
lines.append('-' * 80)

species_info = {}
for sp in SPECIES:
    w      = np.load(os.path.join(DATA_DIR, sp, 'conn.npy'))
    labels = np.load(os.path.join(DATA_DIR, sp, 'labels.npy'))
    n      = w.shape[0]
    edges  = int(np.count_nonzero(w))
    dens   = edges / (n * (n - 1))
    sym    = bool(np.allclose(w, w.T))
    is_bin = bool(np.all((w == 0) | (w == 1)))
    wvals  = w[w > 0]
    n_in   = int(np.sum(labels == 1))
    species_info[sp] = {
        'n': n, 'edges': edges, 'density': dens, 'symmetric': sym,
        'binary': is_bin, 'w_min': float(wvals.min()), 'w_max': float(wvals.max()),
        'n_input': n_in
    }
    lines.append(f'{sp:<12} {n:>6} {edges:>8} {dens:>9.4f} '
                 f'{str(sym):>10} {str(is_bin):>8} {wvals.min():>8.4f} '
                 f'{wvals.max():>8.2f} {n_in:>8}')

lines.append('')
lines.append('Sources:')
lines.append('  fly     — Chiang et al. 2011,    Drosophila brain,   49 regions')
lines.append('  mouse   — Rubinov et al. 2015,   mouse cortex,      112 regions')
lines.append('  rat     — Bota et al. 2015,       rat cortex,         73 regions')
lines.append('  macaque — Modha & Singh 2010,    macaque brain,     242 regions')
lines.append('')
lines.append('Input node assignment:')
lines.append('  fly     — olfactory network regions (biologically motivated)')
lines.append('  mouse/rat/macaque — random 20% (seed=42); sensitivity analysis')
lines.append('  across 4 seeds confirms CV < 15% for all species (see input_sensitivity.csv)')

# ── hyperparameters ───────────────────────────────────────────────────────────
lines.append(_sec('HYPERPARAMETERS'))
lines.append(textwrap.dedent("""
ESN / Reservoir:
  Spectral radius (ρ)     = 0.97    [MC increases as ρ→1; Dambre et al. 2012]
  Washout steps           = 200     [standard ESN practice]
  Signal length (N_SAMPLES) = 2000
  Max lag (MC)            = 50
  Train/test split        = 70/30
  Readout regularisation  = RidgeCV (baselines) / Ridge α=1.0 (PSO inner loop)
  Input weight w_in       = 1/n_input (uniform, only input-labelled nodes)

PSO:
  Particles (N)           = 20
  Iterations              = 50     [1000 evaluations per run]
  Inertia w               = 0.7
  Cognitive c1            = 2.0
  Social c2               = 2.0    [Kennedy & Eberhart 1995 standard]
  Weight bound            = [0, bio_max × 3.0]
  Initialisation A        = Gaussian(bio_weights, 0.3·σ_bio)
  Initialisation B        = Uniform(0, max_w)
  PSO runs per condition  = 5      [independent runs, different init seeds]

Seeds:
  Optimisation signal     = seed = run index (0..4)
  Held-out eval signal    = seed = run + 100  [prevents selection bias]
  PSO init RNG            = seed = run*997+31 [separate from signal seed]
""").strip())

# ── baseline results ──────────────────────────────────────────────────────────
base_df = _load('baseline_results.csv')
if base_df is not None:
    lines.append(_sec('BIOLOGICAL BASELINE RESULTS (Step 02)'))
    lines.append('Memory Capacity (MC = Σ R² over 50 lags):')
    lines.append(f'  {"Species":<10} {"Mean":>8} {"SD":>8} {"Min":>8} {"Max":>8}')
    lines.append('  ' + '-' * 42)
    for sp in SPECIES:
        d = base_df[base_df.species == sp]['mc']
        lines.append(f'  {sp:<10} {d.mean():>8.3f} {d.std():>8.3f} '
                     f'{d.min():>8.3f} {d.max():>8.3f}')

    lines.append('')
    lines.append('Lorenz NRMSE (lower = better):')
    lines.append(f'  {"Species":<10} {"Mean":>8} {"SD":>8} {"Min":>8} {"Max":>8}')
    lines.append('  ' + '-' * 42)
    for sp in SPECIES:
        d = base_df[base_df.species == sp]['lorenz_nrmse']
        lines.append(f'  {sp:<10} {d.mean():>8.4f} {d.std():>8.4f} '
                     f'{d.min():>8.4f} {d.max():>8.4f}')

# ── PSO results ───────────────────────────────────────────────────────────────
pso_df = _load('pso_results.csv')
if pso_df is not None:
    lines.append(_sec('PSO OPTIMISATION RESULTS — HELD-OUT EVALUATION (Step 03)'))
    lines.append('All scores evaluated on held-out signal (seed = run + 100).')
    lines.append('PSO was optimised on a separate signal (seed = run).')
    lines.append('')

    for sp in SPECIES:
        df_sp = pso_df[pso_df.species == sp]
        lines.append(_sub(f'{sp.upper()}'))
        for cond, col in [('Biological', 'mc_bio'), ('PSO-A (from bio)', 'mc_pso_a'),
                          ('PSO-B (from random)', 'mc_pso_b'), ('Random null', 'mc_random')]:
            d = df_sp[col]
            lines.append(f'  {cond:<22}: {d.mean():.3f} ± {d.std():.3f}  '
                         f'[{d.min():.3f} – {d.max():.3f}]  N={len(d)}')

        # PSO-A vs B optimisation scores (for convergence context)
        if 'mc_pso_a_opt' in df_sp.columns:
            lines.append('')
            lines.append('  Optimisation-signal scores (not for comparison, convergence ref):')
            for cond, col in [('PSO-A opt', 'mc_pso_a_opt'), ('PSO-B opt', 'mc_pso_b_opt')]:
                if col in df_sp.columns:
                    d = df_sp[col]
                    lines.append(f'    {cond:<12}: {d.mean():.3f} ± {d.std():.3f}')

# ── statistical tests ─────────────────────────────────────────────────────────
stats_df = _load('stats_summary.csv')
if stats_df is not None:
    lines.append(_sec('STATISTICAL TESTS (Paired t-test, df=4)'))
    lines.append('Note: N=5 gives df=4 (low power). Cohen\'s d provided for effect size.')
    lines.append('Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant')
    lines.append('')
    lines.append(f'{"Species":<10} {"Condition":<20} {"Bio μ":>7} {"Cond μ":>8} '
                 f'{"Δ%":>7} {"t":>6} {"p":>9} {"d":>6} {"sig":>5}')
    lines.append('-' * 80)
    for _, row in stats_df.iterrows():
        lines.append(
            f'{row.species:<10} {row.condition:<20} {row.bio_mean:>7.3f} '
            f'{row.cond_mean:>8.3f} {row.delta_pct:>6.1f}% '
            f'{row.t_stat:>6.2f} {row.p_value:>9.4f} {row.cohens_d:>6.2f} '
            f'{row.significance:>5}')
    lines.append('')
    lines.append('Interpretation guide for Cohen\'s d:')
    lines.append('  small ≈ 0.2, medium ≈ 0.5, large ≈ 0.8, very large > 1.2')

# ── graph metrics ─────────────────────────────────────────────────────────────
graph_df = _load('graph_metrics.csv')
if graph_df is not None:
    lines.append(_sec('GRAPH TOPOLOGY METRICS (Weighted Directed, Fagiolo 2007)'))
    lines.append('Clustering: directed weighted (Fagiolo 2007)')
    lines.append('Path length: shortest path with distance = 1/weight (weighted distance)')
    lines.append('  → Differs across conditions because PSO changes weights.')
    lines.append('  → Binary path length would be IDENTICAL (topology unchanged by PSO).')
    lines.append('σ: directed small-world index; mean_k = edges/nodes (directed formula)')
    lines.append('')
    lines.append(f'{"Species":<10} {"Condition":<10} {"Clustering":>12} '
                 f'{"Path(1/w)":>12} {"σ":>8}')
    lines.append('-' * 56)
    for _, row in graph_df.iterrows():
        bin_flag = ' [binary]' if row.get('is_binary_source', False) and row.condition == 'Bio' else ''
        lines.append(f'{row.species:<10} {row.condition:<10} '
                     f'{row.clustering:>12.4f} {row.path_length:>12.3f} '
                     f'{row.sigma:>8.3f}{bin_flag}')

# ── input sensitivity ─────────────────────────────────────────────────────────
sens_df = _load('input_sensitivity.csv')
if sens_df is not None:
    lines.append(_sec('INPUT NODE SENSITIVITY ANALYSIS'))
    lines.append('Tests whether baseline MC depends on which 20% of nodes receive input.')
    lines.append('Seeds 42 (default), 7, 13, 99 all use random 20% of non-fly species.')
    lines.append('')
    lines.append(f'{"Species":<10} {"Mean MC":>9} {"SD":>7} {"CV%":>7} {"Seeds tested"}')
    lines.append('-' * 50)
    for sp in sens_df['species'].unique():
        d = sens_df[sens_df.species == sp]['mc']
        cv = d.std() / (d.mean() + 1e-10) * 100
        seeds = list(sens_df[sens_df.species == sp]['input_seed'])
        lines.append(f'{sp:<10} {d.mean():>9.3f} {d.std():>7.3f} {cv:>6.1f}%  {seeds}')

# ── figures ───────────────────────────────────────────────────────────────────
lines.append(_sec('FIGURES'))
figures = [
    ('01_matrix_heatmaps.png',    'Connectome weight matrices (log-scale for fly). '
                                   'Row=source, col=target. Fly uses log1p to handle '
                                   'high synapse counts (up to ~4500).'),
    ('02_degree_distributions.png','Out-degree (blue) and in-degree (orange) histograms '
                                   'for all 4 species. Reveals scale-free tendencies.'),
    ('03_mc_baselines.png',        'Biological baseline Memory Capacity per species. '
                                   'Bar = mean ± SD, N=5 independent signal seeds. '
                                   'ρ=0.97 (Dambre 2012 standard).'),
    ('04_lorenz_baselines.png',    'Biological baseline Lorenz one-step NRMSE per species '
                                   '(lower = better). N=5 independent initial conditions.'),
    ('05_pso_convergence.png',     'PSO convergence over 50 iterations. All 5 runs shown '
                                   'in light colour; best run highlighted bold. '
                                   'PSO-A=green, PSO-B=orange. Bio baseline (dashed blue) '
                                   'is the held-out mean for context.'),
    ('06_mc_boxplots.png',         'MC scores on HELD-OUT signal: Bio (blue), PSO-A (green), '
                                   'PSO-B (orange), Random (grey). Significance brackets '
                                   'show *** / ** / * / ns from paired t-test. '
                                   'ns labels also show Cohen\'s d.'),
    ('07_graph_metrics.png',       'Weighted directed graph metrics: clustering coefficient '
                                   '(Fagiolo 2007), weighted path length (1/w distance), '
                                   'small-world index σ. Metrics differ across conditions '
                                   'because PSO changes edge weights. Binary path length '
                                   'would be identical (topology unchanged).'),
]

for fname, desc in figures:
    fpath = os.path.join(IMAGES_DIR, fname)
    exists = '✓' if os.path.exists(fpath) else '✗ (not yet generated)'
    lines.append(f'[{exists}] {fname}')
    for l in textwrap.wrap(desc, width=66):
        lines.append(f'      {l}')
    lines.append('')

# ── reviewer response summary ─────────────────────────────────────────────────
lines.append(_sec('METHODOLOGICAL NOTES — REVIEWER 2 RESPONSES'))
notes = [
    ('CRITICAL #1 — Figure 07 graph metrics',
     'FIXED. Binary path length/sigma would be identical across Bio/PSO-A/PSO-B '
     'because PSO preserves topology. Step04 now uses weighted directed metrics: '
     'clustering (Fagiolo 2007 DiGraph), path length (1/w as distance), directed σ.'),
    ('CRITICAL #2 — Selection bias in PSO evaluation',
     'FIXED. PSO optimises on signal A (seed=run); all reported scores use held-out '
     'signal B (seed=run+100). Bio, PSO-A, PSO-B, Random all evaluated on B.'),
    ('CRITICAL #3 — Input node assignment',
     'ACKNOWLEDGED + SENSITIVITY ANALYSIS ADDED. Fly uses biological olfactory nodes. '
     'Mouse/rat/macaque use random 20% (seed=42, limited by dataset metadata). '
     'Sensitivity analysis across 4 seeds shows CV < 15% for all species '
     '(see input_sensitivity.csv). Biologically motivated assignment is future work.'),
    ('MAJOR #4 — N=5 low statistical power',
     'ACKNOWLEDGED. Cohen\'s d added to all tests. Results reported with both p-values '
     'and effect sizes. Text notes df=4 limitation. N=5 is computationally constrained.'),
    ('MAJOR #5 — Spectral radius 0.9',
     'FIXED. Changed to ρ=0.97 in both step02 and step03 (Dambre et al. 2012).'),
    ('MAJOR #6 — Ridge α=1e-6 unregularised',
     'FIXED. Baselines use RidgeCV with 5-fold CV over {0.01, 0.1, 1.0, 10, 100}. '
     'PSO inner loop uses Ridge(α=1.0) — CV in inner loop is prohibitively slow.'),
    ('MAJOR #7 — PSO convergence insufficient',
     'IMPROVED. N_ITERATIONS: 30→50 (600→1000 evaluations/run). '
     'Convergence plots now show all 5 runs to make plateau assessment transparent.'),
    ('MAJOR #8 — Macaque binary connectome',
     'FLAGGED. Macaque (Modha & Singh 2010) has binary connectivity. PSO treating it '
     'as continuous is an explicit design choice (testing whether heterogeneous '
     'weighting improves MC over uniform binary). Noted in all relevant outputs.'),
    ('MAJOR #9 — Directed graph clustering on undirected graph',
     'FIXED. All metrics use nx.DiGraph. mean_k = n_edges/n_nodes (directed formula, '
     'no factor 2). Clustering uses nx.average_clustering with weight on DiGraph.'),
    ('MINOR #10 — Power iteration 25 steps',
     'FIXED. Increased to 50 iterations in _batch_spectral_radius.'),
    ('MINOR #12 — Convergence cherry-picked',
     'FIXED. All 5 runs shown in light colour; best run highlighted bold.'),
    ('MINOR #13 — Seed conflation',
     'FIXED. PSO init RNG: seed=run*997+31 (distinct from signal seeds).'),
]

for title, response in notes:
    lines.append(f'\n{title}:')
    for l in textwrap.wrap(response, width=68):
        lines.append(f'  {l}')

# ── write file ────────────────────────────────────────────────────────────────
with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(lines))
    f.write('\n')

print(f'Saved: {OUT_FILE}  ({len(lines)} lines)')
print('\nStep 05 complete.')
