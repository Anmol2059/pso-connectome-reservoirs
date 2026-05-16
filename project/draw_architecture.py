"""
Architecture diagrams for the Wild Reservoirs paper.
Saves two publication-ready figures to the Overleaf images/ directory.

  arch_a_pipeline.png  — ESN pipeline: connectome → reservoir → readout → tasks
  arch_b_overview.png  — Experimental design: species × algorithms × tasks grid
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ── output path ───────────────────────────────────────────────────────────────
PROJ_DIR    = os.path.dirname(os.path.abspath(__file__))
OVERLEAF    = os.path.join(os.path.dirname(PROJ_DIR), 'overleaf-pso-paper', 'images')
os.makedirs(OVERLEAF, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
})

# ── colour palette ─────────────────────────────────────────────────────────────
C = {
    'species':   ('#E3F2FD', '#1565C0'),   # face, edge
    'reservoir': ('#EDE7F6', '#4527A0'),
    'optimizer': ('#FFF8E1', '#E65100'),
    'readout':   ('#FCE4EC', '#880E4F'),
    'task':      ('#E8F5E9', '#1B5E20'),
    'input':     ('#F1F8E9', '#33691E'),
    'arrow':     '#546E7A',
    'bg':        '#FAFAFA',
}

def fancy_box(ax, x, y, w, h, fc, ec, text, fontsize=9, bold=False,
              text2=None, text2_size=7.5, radius=0.015):
    """Draw a rounded box with optional two-line text."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f'round,pad=0,rounding_size={radius}',
                         facecolor=fc, edgecolor=ec, linewidth=1.6,
                         transform=ax.transAxes, clip_on=False, zorder=3)
    # subtle drop shadow
    shadow = FancyBboxPatch((x + 0.003, y - 0.003), w, h,
                            boxstyle=f'round,pad=0,rounding_size={radius}',
                            facecolor='#CCCCCC', edgecolor='none',
                            transform=ax.transAxes, clip_on=False, zorder=2,
                            alpha=0.45)
    ax.add_patch(shadow)
    ax.add_patch(box)

    cx, cy = x + w / 2, y + h / 2
    if text2:
        cy += h * 0.12
    ax.text(cx, cy, text,
            ha='center', va='center', transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold' if bold else 'normal',
            zorder=4)
    if text2:
        ax.text(cx, y + h * 0.30, text2,
                ha='center', va='center', transform=ax.transAxes,
                fontsize=text2_size, color='#444444', zorder=4,
                style='italic')


def arrow(ax, x0, y0, x1, y1, label='', color='#546E7A', lw=1.5,
          arrowstyle='->', mutation_scale=14):
    """Draw a clean arrow between two points (axes coordinates)."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=mutation_scale),
                zorder=5)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.025, label, ha='center', va='bottom',
                transform=ax.transAxes, fontsize=7.5, color=color,
                fontweight='bold', zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — ESN Pipeline Architecture
# ══════════════════════════════════════════════════════════════════════════════
print('Drawing Figure A: Pipeline architecture...')

fig_a, ax_a = plt.subplots(figsize=(13, 7))
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
ax_a.axis('off')
fig_a.patch.set_facecolor(C['bg'])
ax_a.set_facecolor(C['bg'])

# ── title ────────────────────────────────────────────────────────────────────
ax_a.text(0.5, 0.965, 'Connectome Reservoir Computing — Experimental Pipeline',
          ha='center', va='top', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', color='#212121')
ax_a.plot([0, 1], [0.935, 0.935], color='#BDBDBD', lw=1.0, transform=ax_a.transAxes)

# ── INPUT SIGNAL box ─────────────────────────────────────────────────────────
fancy_box(ax_a, 0.01, 0.36, 0.13, 0.28,
          C['input'][0], C['input'][1],
          '📥  Input Signal\nu(t)',
          fontsize=9.5, bold=True,
          text2='u(t) ~ U(−1, 1)  [MC]\nLorenz x(t)  [Lorenz]\nNARMA u(t)  [NARMA]\nMG x(t)  [MG]',
          text2_size=7.2)

# mini time-series sketch inside input box
t_sketch = np.linspace(0, 1, 60)
sig_sketch = 0.055 * np.sin(2 * np.pi * 5 * t_sketch) + 0.015 * np.random.default_rng(1).standard_normal(60)
ax_a.plot(0.015 + t_sketch * 0.12, 0.275 + sig_sketch,
          color=C['input'][1], lw=0.9, transform=ax_a.transAxes, zorder=5, alpha=0.7)

arrow(ax_a, 0.140, 0.50, 0.205, 0.50, label='u(t)')

# ── CONNECTOME / RESERVOIR box ────────────────────────────────────────────────
fancy_box(ax_a, 0.205, 0.18, 0.28, 0.64,
          C['reservoir'][0], C['reservoir'][1],
          '🧠  Connectome Reservoir',
          fontsize=11, bold=True, radius=0.018)

# Draw a mini network inside the reservoir box
rng = np.random.default_rng(42)
n_nodes = 14
theta_nodes = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
nx_ = 0.345 + 0.085 * np.cos(theta_nodes)
ny_ = 0.620 + 0.10 * np.sin(theta_nodes)
# edges
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        if rng.random() < 0.22:
            ax_a.plot([nx_[i], nx_[j]], [ny_[i], ny_[j]],
                      color='#7E57C2', lw=0.7, alpha=0.55,
                      transform=ax_a.transAxes, zorder=4)
# nodes coloured by role
node_colors = ['#00897B' if i < 3 else '#EF5350' if i > 10 else '#9C27B0'
               for i in range(n_nodes)]
for i in range(n_nodes):
    ax_a.plot(nx_[i], ny_[i], 'o', color=node_colors[i],
              ms=5.5, transform=ax_a.transAxes, zorder=5,
              markeredgewidth=0.6, markeredgecolor='white')

# State update equation
ax_a.text(0.345, 0.495,
          r'$\mathbf{x}(t)=\tanh\!\left(\mathbf{W}\mathbf{x}(t{-}1)'
          r'+\mathbf{W}_{in}\mathbf{u}(t)\right)$',
          ha='center', va='center', transform=ax_a.transAxes,
          fontsize=8.8, color='#311B92', zorder=6,
          bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#7E57C2',
                    alpha=0.90, lw=1.0))

ax_a.text(0.345, 0.405,
          r'$\rho(\mathbf{W}) = 0.97$   •   Washout = 200 steps',
          ha='center', va='center', transform=ax_a.transAxes,
          fontsize=8, color='#4527A0', zorder=6, style='italic')

# Sparsity note
ax_a.text(0.345, 0.260,
          '✦  Biological sparsity pattern frozen\n'
          '✦  Only edge weights optimised',
          ha='center', va='center', transform=ax_a.transAxes,
          fontsize=7.8, color='#4527A0', zorder=6,
          linespacing=1.6)

# ── OPTIMISER box (above reservoir) ──────────────────────────────────────────
fancy_box(ax_a, 0.215, 0.855, 0.26, 0.075,
          C['optimizer'][0], C['optimizer'][1],
          '🐦 PSO   🧬 DE   🐺 GWO   🐋 WOA',
          fontsize=9.5, bold=True, radius=0.012)
ax_a.text(0.345, 0.843, 'Bio-inspired gradient-free optimisers  •  1000 evals/run  •  Bio-A init',
          ha='center', va='top', transform=ax_a.transAxes,
          fontsize=7.5, color='#BF360C', style='italic', zorder=4)

# Arrow: optimiser tunes reservoir (curved down)
ax_a.annotate('', xy=(0.345, 0.820), xytext=(0.345, 0.855),
              xycoords='axes fraction', textcoords='axes fraction',
              arrowprops=dict(arrowstyle='->', color=C['optimizer'][1],
                              lw=1.8, mutation_scale=14,
                              connectionstyle='arc3,rad=0'),
              zorder=6)
ax_a.text(0.390, 0.836, 'tune W', ha='left', va='center',
          transform=ax_a.transAxes, fontsize=7.5,
          color=C['optimizer'][1], fontweight='bold')

# Loop arrow showing held-out eval
ax_a.annotate('', xy=(0.220, 0.600), xytext=(0.220, 0.750),
              xycoords='axes fraction', textcoords='axes fraction',
              arrowprops=dict(arrowstyle='<->', color='#78909C',
                              lw=1.2, mutation_scale=10,
                              connectionstyle='arc3,rad=-0.4'),
              zorder=6)
ax_a.text(0.175, 0.675, 'held-out\neval', ha='center', va='center',
          transform=ax_a.transAxes, fontsize=6.8, color='#546E7A',
          style='italic')

arrow(ax_a, 0.485, 0.50, 0.545, 0.50, label='x(t)')

# ── READOUT box ───────────────────────────────────────────────────────────────
fancy_box(ax_a, 0.545, 0.355, 0.135, 0.290,
          C['readout'][0], C['readout'][1],
          '📐  Readout\n(Ridge Regression)',
          fontsize=9, bold=True,
          text2='W_out trained\non 70% of states\nα = 1.0 (opt loop)\nRidgeCV (baseline)',
          text2_size=7.2)

arrow(ax_a, 0.680, 0.50, 0.735, 0.50, label='ŷ(t)')

# ── TASKS column ──────────────────────────────────────────────────────────────
task_data = [
    ('📊', 'Memory Capacity', 'MC = Σ r²(û(t−k), u(t−k))', 0.735),
    ('🌀', 'Lorenz NRMSE',    'one-step-ahead chaotic pred.', 0.735),
    ('📈', 'NARMA-10 NRMSE',  'nonlinear system ID (order 10)', 0.735),
    ('🔮', 'Mackey–Glass NRMSE', 'delay-diff. eq., τ = 17', 0.735),
]

task_y_positions = [0.72, 0.535, 0.35, 0.165]
task_h = 0.145

for (emoji, name, desc, _), y0 in zip(task_data, task_y_positions):
    fancy_box(ax_a, 0.735, y0, 0.255, task_h,
              C['task'][0], C['task'][1],
              f'{emoji}  {name}',
              fontsize=8.8, bold=True,
              text2=desc, text2_size=7.2, radius=0.012)
    arrow(ax_a, 0.680, y0 + task_h / 2, 0.735, y0 + task_h / 2)

# ── SPECIES column (left strip) ──────────────────────────────────────────────
species_data = [
    ('🪱', 'C. elegans', '279 nodes'),
    ('🪰', 'Fly',        '49 nodes'),
    ('🐭', 'Mouse',      '112 nodes'),
    ('🐀', 'Rat',        '73 nodes'),
    ('🐒', 'Macaque',    '29 nodes'),
    ('🧠', 'Human',      '83 nodes'),
]

# Vertical stack left of input box
sp_h   = 0.095
sp_gap = 0.015
sp_y0  = 1.0 - len(species_data) * (sp_h + sp_gap) - 0.07

ax_a.text(0.075, sp_y0 - 0.025, '6 Species', ha='center', va='top',
          transform=ax_a.transAxes, fontsize=8, fontweight='bold',
          color=C['species'][1])

for i, (emoji, name, nodes) in enumerate(species_data):
    y = sp_y0 - i * (sp_h + sp_gap) - sp_h
    # small colored tag
    box_sp = FancyBboxPatch((0.01, y), 0.13, sp_h,
                            boxstyle='round,pad=0,rounding_size=0.010',
                            facecolor=C['species'][0], edgecolor=C['species'][1],
                            linewidth=1.2, transform=ax_a.transAxes, zorder=3)
    ax_a.add_patch(box_sp)
    ax_a.text(0.075, y + sp_h * 0.62, f'{emoji} {name}',
              ha='center', va='center', transform=ax_a.transAxes,
              fontsize=8, fontweight='bold', color='#0D47A1', zorder=4)
    ax_a.text(0.075, y + sp_h * 0.22, nodes,
              ha='center', va='center', transform=ax_a.transAxes,
              fontsize=7, color='#1565C0', zorder=4, style='italic')

# Arrow from species strip to input box
ax_a.annotate('', xy=(0.14, 0.36), xytext=(0.14, 0.10),
              xycoords='axes fraction', textcoords='axes fraction',
              arrowprops=dict(arrowstyle='<|-', color=C['species'][1],
                              lw=1.3, mutation_scale=12),
              zorder=5)

# ── footer ────────────────────────────────────────────────────────────────────
ax_a.text(0.5, 0.025,
          'Bio-A initialisation: optimiser starts from biological weights  •  '
          'Held-out evaluation: train signal ≠ test signal  •  '
          'Sparsity preserved: only edge weights optimised',
          ha='center', va='bottom', transform=ax_a.transAxes,
          fontsize=7.5, color='#616161', style='italic')

plt.tight_layout(pad=0.3)
out_a = os.path.join(OVERLEAF, 'arch_a_pipeline.png')
fig_a.savefig(out_a, dpi=200, bbox_inches='tight', facecolor=C['bg'])
plt.close(fig_a)
print(f'  Saved: {out_a}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Experimental Overview: Species × Algorithms × Tasks
# ══════════════════════════════════════════════════════════════════════════════
print('Drawing Figure B: Experimental overview...')

fig_b, ax_b = plt.subplots(figsize=(12, 8))
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
ax_b.axis('off')
fig_b.patch.set_facecolor(C['bg'])
ax_b.set_facecolor(C['bg'])

ax_b.text(0.5, 0.970, 'Experimental Design Overview',
          ha='center', va='top', transform=ax_b.transAxes,
          fontsize=13, fontweight='bold', color='#212121')
ax_b.plot([0, 1], [0.945, 0.945], color='#BDBDBD', lw=1.0, transform=ax_b.transAxes)

# ── column headers ────────────────────────────────────────────────────────────
for x, label, col in [(0.08, 'SPECIES', C['species'][1]),
                       (0.43, 'ALGORITHMS', C['optimizer'][1]),
                       (0.77, 'BENCHMARK TASKS', C['task'][1])]:
    ax_b.text(x, 0.920, label, ha='center', va='top',
              transform=ax_b.transAxes, fontsize=10,
              fontweight='bold', color=col)

# ── species column ────────────────────────────────────────────────────────────
species_info = [
    ('🪱', 'C. elegans',  '279 nodes  ·  2,194 edges',   'Nematode worm',         '#E3F2FD', '#1565C0'),
    ('🪰', 'Fly',         '49 nodes  ·  1,950 edges',    'Drosophila melanogaster','#E8EAF6', '#283593'),
    ('🐭', 'Mouse',       '112 nodes  ·  6,542 edges',   'Mus musculus cortex',    '#F3E5F5', '#6A1B9A'),
    ('🐀', 'Rat',         '73 nodes  ·  1,923 edges',    'Rattus norvegicus',      '#EDE7F6', '#4527A0'),
    ('🐒', 'Macaque',     '29 nodes  ·  590 edges',      'FLNe weighted (Markov)', '#E8F5E9', '#1B5E20'),
    ('🧠', 'Human',       '83 nodes  ·  2,134 edges',    'DTI tractography (MRI)', '#E0F7FA', '#006064'),
]

n_sp    = len(species_info)
sp_h    = 0.095
sp_gap  = 0.018
y_start = 0.87

for i, (emoji, name, nodes, desc, fc, ec) in enumerate(species_info):
    y = y_start - i * (sp_h + sp_gap)
    box = FancyBboxPatch((0.01, y - sp_h), 0.145, sp_h,
                         boxstyle='round,pad=0,rounding_size=0.012',
                         facecolor=fc, edgecolor=ec, linewidth=1.4,
                         transform=ax_b.transAxes, zorder=3)
    shadow = FancyBboxPatch((0.013, y - sp_h - 0.004), 0.145, sp_h,
                            boxstyle='round,pad=0,rounding_size=0.012',
                            facecolor='#CCCCCC', edgecolor='none',
                            transform=ax_b.transAxes, zorder=2, alpha=0.4)
    ax_b.add_patch(shadow); ax_b.add_patch(box)
    ax_b.text(0.083, y - sp_h * 0.35, f'{emoji}  {name}',
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=9.5, fontweight='bold', color=ec, zorder=4)
    ax_b.text(0.083, y - sp_h * 0.70, nodes,
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=7.2, color='#424242', zorder=4)

# ── algorithms column ─────────────────────────────────────────────────────────
alg_info = [
    ('🐦', 'PSO',  'Particle Swarm\nOptimisation',  'Kennedy & Eberhart 1995', '#FFF8E1', '#F57F17'),
    ('🧬', 'DE',   'Differential\nEvolution',       'Storn & Price 1997',      '#FBE9E7', '#BF360C'),
    ('🐺', 'GWO',  'Grey Wolf\nOptimiser',          'Mirjalili et al. 2014',   '#F3E5F5', '#6A1B9A'),
    ('🐋', 'WOA',  'Whale Optimisation\nAlgorithm', 'Mirjalili & Lewis 2016',  '#E3F2FD', '#1565C0'),
]

alg_h   = 0.120
alg_gap = 0.025
alg_y0  = y_start

for i, (emoji, short, name, ref, fc, ec) in enumerate(alg_info):
    y = alg_y0 - i * (alg_h + alg_gap)
    box = FancyBboxPatch((0.165, y - alg_h), 0.165, alg_h,
                         boxstyle='round,pad=0,rounding_size=0.012',
                         facecolor=fc, edgecolor=ec, linewidth=1.4,
                         transform=ax_b.transAxes, zorder=3)
    shadow = FancyBboxPatch((0.168, y - alg_h - 0.004), 0.165, alg_h,
                            boxstyle='round,pad=0,rounding_size=0.012',
                            facecolor='#CCCCCC', edgecolor='none',
                            transform=ax_b.transAxes, zorder=2, alpha=0.4)
    ax_b.add_patch(shadow); ax_b.add_patch(box)
    ax_b.text(0.248, y - alg_h * 0.28, f'{emoji}  {short}',
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=10.5, fontweight='bold', color=ec, zorder=4)
    ax_b.text(0.248, y - alg_h * 0.57, name,
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=7.8, color='#424242', zorder=4, linespacing=1.4)
    ax_b.text(0.248, y - alg_h * 0.83, ref,
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=6.8, color='#757575', zorder=4, style='italic')

# ── central ESN box ───────────────────────────────────────────────────────────
fancy_box(ax_b, 0.345, 0.275, 0.185, 0.580,
          C['reservoir'][0], C['reservoir'][1],
          '🔬  ESN  Reservoir',
          fontsize=11, bold=True, radius=0.016)

# small equation
ax_b.text(0.437, 0.620,
          r'$\mathbf{x}(t)=\tanh(\mathbf{W}\mathbf{x}(t{-}1)+\mathbf{W}_{in}\mathbf{u}(t))$',
          ha='center', va='center', transform=ax_b.transAxes,
          fontsize=7.8, color='#311B92', zorder=5,
          bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#7E57C2',
                    alpha=0.9, lw=0.8))

# mini network
rng2   = np.random.default_rng(7)
n_mini = 12
th2    = np.linspace(0, 2 * np.pi, n_mini, endpoint=False)
nx2    = 0.437 + 0.068 * np.cos(th2)
ny2    = 0.500 + 0.085 * np.sin(th2)
for i in range(n_mini):
    for j in range(i + 1, n_mini):
        if rng2.random() < 0.18:
            ax_b.plot([nx2[i], nx2[j]], [ny2[i], ny2[j]],
                      color='#7E57C2', lw=0.7, alpha=0.5,
                      transform=ax_b.transAxes, zorder=4)
nc2 = ['#00897B' if i < 2 else '#EF5350' if i > 9 else '#9C27B0' for i in range(n_mini)]
for i in range(n_mini):
    ax_b.plot(nx2[i], ny2[i], 'o', color=nc2[i], ms=5,
              transform=ax_b.transAxes, zorder=5,
              markeredgewidth=0.5, markeredgecolor='white')

ax_b.text(0.437, 0.370,
          r'$\rho(\mathbf{W})=0.97$  ·  washout=200',
          ha='center', va='center', transform=ax_b.transAxes,
          fontsize=7.5, color='#4527A0', style='italic', zorder=5)
ax_b.text(0.437, 0.315,
          'Ridge Readout  (W_out trained)',
          ha='center', va='center', transform=ax_b.transAxes,
          fontsize=7.5, color='#880E4F', fontweight='bold', zorder=5)

# ── task column ───────────────────────────────────────────────────────────────
task_info = [
    ('📊', 'Memory Capacity',  'MC = Σ r²(û(t−k), u(t−k))',  'k = 1..50  ↑ higher = better', '#DCEDC8', '#33691E'),
    ('🌀', 'Lorenz Attractor', 'NRMSE one-step-ahead',        'σ=10, ρ=28, β=8/3  ↓ lower = better', '#FFE0B2', '#BF360C'),
    ('📈', 'NARMA-10',         'System identification NRMSE', 'order-10 nonlinear  ↓ lower = better', '#F8BBD9', '#880E4F'),
    ('🔮', 'Mackey–Glass',     'Chaotic forecast NRMSE',      'τ=17, dt=0.1  ↓ lower = better',       '#B2EBF2', '#006064'),
]

task_h2  = 0.150
task_gap = 0.022
task_y0  = y_start

for i, (emoji, name, metric, note, fc, ec) in enumerate(task_info):
    y = task_y0 - i * (task_h2 + task_gap)
    box = FancyBboxPatch((0.550, y - task_h2), 0.195, task_h2,
                         boxstyle='round,pad=0,rounding_size=0.012',
                         facecolor=fc, edgecolor=ec, linewidth=1.4,
                         transform=ax_b.transAxes, zorder=3)
    shadow = FancyBboxPatch((0.553, y - task_h2 - 0.004), 0.195, task_h2,
                            boxstyle='round,pad=0,rounding_size=0.012',
                            facecolor='#CCCCCC', edgecolor='none',
                            transform=ax_b.transAxes, zorder=2, alpha=0.4)
    ax_b.add_patch(shadow); ax_b.add_patch(box)
    ax_b.text(0.648, y - task_h2 * 0.25, f'{emoji}  {name}',
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=9.5, fontweight='bold', color=ec, zorder=4)
    ax_b.text(0.648, y - task_h2 * 0.55, metric,
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=7.5, color='#424242', zorder=4)
    ax_b.text(0.648, y - task_h2 * 0.82, note,
              ha='center', va='center', transform=ax_b.transAxes,
              fontsize=6.8, color='#757575', zorder=4, style='italic')

# ── arrows between columns ────────────────────────────────────────────────────
# Species → ESN
for i in range(n_sp):
    y_sp = y_start - i * (sp_h + sp_gap) - sp_h / 2
    if 0.28 < y_sp < 0.85:
        ax_b.annotate('', xy=(0.345, y_sp), xytext=(0.310, y_sp),
                      xycoords='axes fraction', textcoords='axes fraction',
                      arrowprops=dict(arrowstyle='->', color=C['species'][1],
                                      lw=1.1, mutation_scale=9), zorder=5)

# Alg → ESN
for i in range(len(alg_info)):
    y_alg = alg_y0 - i * (alg_h + alg_gap) - alg_h / 2
    if 0.28 < y_alg < 0.85:
        ax_b.annotate('', xy=(0.345, y_alg), xytext=(0.330, y_alg),
                      xycoords='axes fraction', textcoords='axes fraction',
                      arrowprops=dict(arrowstyle='->', color=C['optimizer'][1],
                                      lw=1.1, mutation_scale=9), zorder=5)

# ESN → Tasks
for i in range(len(task_info)):
    y_tsk = task_y0 - i * (task_h2 + task_gap) - task_h2 / 2
    if 0.28 < y_tsk < 0.85:
        ax_b.annotate('', xy=(0.550, y_tsk), xytext=(0.530, y_tsk),
                      xycoords='axes fraction', textcoords='axes fraction',
                      arrowprops=dict(arrowstyle='->', color=C['reservoir'][1],
                                      lw=1.1, mutation_scale=9), zorder=5)

# column labels
ax_b.text(0.155, 0.255, '6 species', ha='center', va='center',
          transform=ax_b.transAxes, fontsize=8, color=C['species'][1],
          fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.25', fc=C['species'][0],
                    ec=C['species'][1], lw=0.8))
ax_b.text(0.248, 0.248, '4 algorithms', ha='center', va='center',
          transform=ax_b.transAxes, fontsize=8, color=C['optimizer'][1],
          fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.25', fc=C['optimizer'][0],
                    ec=C['optimizer'][1], lw=0.8))
ax_b.text(0.648, 0.225, '4 tasks', ha='center', va='center',
          transform=ax_b.transAxes, fontsize=8, color=C['task'][1],
          fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.25', fc=C['task'][0],
                    ec=C['task'][1], lw=0.8))

# ── stats strip at bottom ────────────────────────────────────────────────────
ax_b.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.075,
               boxstyle='round,pad=0,rounding_size=0.010',
               facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=1.0,
               transform=ax_b.transAxes, zorder=3))

stats_text = (
    '10 runs per condition   ·   20 particles × 50 iterations = 1000 evals   ·   '
    'Bio-A initialisation from biological weights   ·   '
    'Held-out evaluation (train seed ≠ test seed)   ·   '
    'GPU-batched objective (PyTorch)   ·   Sparsity pattern frozen'
)
ax_b.text(0.50, 0.057, stats_text, ha='center', va='center',
          transform=ax_b.transAxes, fontsize=7.5, color='#37474F',
          zorder=4)

plt.tight_layout(pad=0.3)
out_b = os.path.join(OVERLEAF, 'arch_b_overview.png')
fig_b.savefig(out_b, dpi=200, bbox_inches='tight', facecolor=C['bg'])
plt.close(fig_b)
print(f'  Saved: {out_b}')

print('\nDone. Both architecture figures saved to Overleaf images/.')


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESS: stamp NotoColorEmoji glyphs onto saved PNGs
# ══════════════════════════════════════════════════════════════════════════════
from PIL import Image, ImageDraw, ImageFont as PilFont

EMOJI_FONT_PATH = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'
EMOJI_SIZE      = 109   # NotoColorEmoji requires this fixed size

# Scale factor for stamping: emoji rendered at 109px, scaled to target px
def stamp_emojis(png_path, placements):
    """
    placements: list of (emoji_char, x_pct, y_pct, target_px)
      x_pct/y_pct: 0..1 relative to image, measured from top-left
      target_px: desired rendered size in pixels
    """
    fnt  = PilFont.truetype(EMOJI_FONT_PATH, EMOJI_SIZE)
    base = Image.open(png_path).convert('RGBA')
    W, H = base.size

    for (emoji_char, xp, yp, tpx) in placements:
        # Render emoji onto a transparent tile
        tile = Image.new('RGBA', (EMOJI_SIZE + 20, EMOJI_SIZE + 20), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        d.text((10, 10), emoji_char, font=fnt, embedded_color=True)
        # Scale tile to target size
        scale = tpx / EMOJI_SIZE
        new_w = int(tile.width  * scale)
        new_h = int(tile.height * scale)
        tile  = tile.resize((new_w, new_h), Image.LANCZOS)
        # Paste at position (centred on xp, yp)
        px = int(xp * W) - new_w // 2
        py = int(yp * H) - new_h // 2
        base.paste(tile, (px, py), tile)

    base.convert('RGB').save(png_path, dpi=(200, 200))


# ── Figure A emoji placements (x%, y% from top-left, size px) ────────────────
stamp_emojis(out_a, [
    # Input box
    ('📥', 0.075, 0.530, 22),
    # Reservoir
    ('🧠', 0.345, 0.880, 22),
    # Optimizer
    ('🐦', 0.248, 0.134, 20),
    ('🧬', 0.295, 0.134, 20),
    ('🐺', 0.342, 0.134, 20),
    ('🐋', 0.390, 0.134, 20),
    # Tasks
    ('📊', 0.748, 0.235, 20),
    ('🌀', 0.748, 0.418, 20),
    ('📈', 0.748, 0.600, 20),
    ('🔮', 0.748, 0.783, 20),
    # Species strip
    ('🪱', 0.044, 0.217, 17),
    ('🪰', 0.044, 0.328, 17),
    ('🐭', 0.044, 0.438, 17),
    ('🐀', 0.044, 0.548, 17),
    ('🐒', 0.044, 0.658, 17),
    ('🧠', 0.044, 0.768, 17),
])
print('  Emoji stamped on arch_a_pipeline.png')

# ── Figure B emoji placements ─────────────────────────────────────────────────
stamp_emojis(out_b, [
    # Species column
    ('🪱', 0.083, 0.168, 20),
    ('🪰', 0.083, 0.305, 20),
    ('🐭', 0.083, 0.443, 20),
    ('🐀', 0.083, 0.580, 20),
    ('🐒', 0.083, 0.718, 20),
    ('🧠', 0.083, 0.855, 20),
    # Algorithm column
    ('🐦', 0.248, 0.175, 20),
    ('🧬', 0.248, 0.328, 20),
    ('🐺', 0.248, 0.482, 20),
    ('🐋', 0.248, 0.635, 20),
    # ESN box
    ('🔬', 0.437, 0.130, 22),
    # Tasks column
    ('📊', 0.648, 0.168, 20),
    ('🌀', 0.648, 0.333, 20),
    ('📈', 0.648, 0.498, 20),
    ('🔮', 0.648, 0.663, 20),
])
print('  Emoji stamped on arch_b_overview.png')
print('\nAll done.')
