# 🧠 PSO-Optimised Connectome Reservoirs

> **Can swarm intelligence match 500 million years of evolution?**
> We rewire real biological brain connectomes with PSO and test whether the result
> outperforms — or rediscovers — nature's own solution.

**Course:** Bio-Inspired Learning — UPM, MUTSC &nbsp;|&nbsp; **Format:** IEEE

---

## 🔬 Research Question

Real brains are wired by evolution. PSO is a swarm algorithm that optimises
by collective exploration. This project asks:

1. Can PSO find connectome weights that **match or surpass biological Memory Capacity**?
2. Does PSO, starting from scratch, **rediscover the weighted structure** evolution built?
3. If both succeed — what does that tell us about the computational demands of neural topology?

---

## 🐝 Real Biological Datasets

Connectomes are fetched automatically via `netneurotools.fetch_famous_gmat`.

| Species | Nodes | Edges | Density | Weights | Source |
|---------|------:|------:|--------:|---------|--------|
| 🪰 Fruit fly (*Drosophila*) | 49  | 1 950 | 82.9% | Continuous | Chiang et al. 2011 |
| 🐭 Mouse cortex             | 112 | 6 542 | 52.6% | Continuous | Rubinov et al. 2015 |
| 🐀 Rat cortex               | 73  | 1 923 | 36.6% | Continuous | Bota et al. 2015 |
| 🐒 Macaque                  | 242 | 4 090 |  7.0% | **Binary** | Modha & Singh 2010 |

All connectomes are **directed and asymmetric**. Macaque is binary (presence/absence only);
PSO on macaque is treated as a separate sub-experiment (adding continuous weights to topology-only data).

---

## ⚡ Quick Start

```bash
git clone https://github.com/Anmol2059/pso-connectome-reservoirs.git
cd pso-connectome-reservoirs

# 1. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Install conn2res (local fork, no-deps)
git clone https://github.com/netneurolab/conn2res.git
cd conn2res && pip install --no-deps . && cd ..

# 3. Run the full pipeline
cd project
bash run.sh
```

> 💡 **Apple Silicon (MPS):** PyTorch auto-detects the MPS backend — step 03 runs
> the entire PSO swarm as a single batched `torch.bmm` call, cutting runtime
> significantly vs CPU.

---

## 🗂️ Pipeline

```
bash run.sh
│
├── 00  fetch_real_connectomes.py   Download & preprocess 4 biological connectomes
│                                   → data/{species}/conn.npy + labels.npy
│
├── 01  step01_explore.py           Visualise weight matrices & degree distributions
│                                   → images/01_matrix_heatmaps.png
│                                   → images/02_degree_distributions.png
│
├── 02  step02_baselines.py         MC + Lorenz baselines (5 independent runs, RidgeCV)
│                                   → data/baseline_results.csv
│                                   → images/03_mc_baselines.png
│                                   → images/04_lorenz_baselines.png
│
├── 03  step03_pso.py               PSO optimisation — 3 conditions × 4 species × 5 runs
│                                   held-out evaluation (PSO optimises on signal A,
│                                   all scores reported on held-out signal B)
│                                   → data/pso_results.csv
│                                   → data/{sp}/conn_pso_a.npy + conn_pso_b.npy
│                                   → images/05_pso_convergence.png
│
├── 04  step04_analysis.py          Paired t-tests + Cohen's d, box plots,
│                                   weighted directed graph metrics (Fagiolo 2007),
│                                   input-node sensitivity analysis
│                                   → data/stats_summary.csv
│                                   → data/graph_metrics.csv
│                                   → data/input_sensitivity.csv
│                                   → images/06_mc_boxplots.png
│                                   → images/07_graph_metrics.png
│
└── 05  step05_summary.py           Compile everything → data/summary.txt
                                    (LLM-ready: all results, methods, reviewer
                                     responses, figure descriptions in one file)
```

Each step logs to `logs/step0X.log` and can be re-run independently.

---

## ⏱️ Expected Runtime

| Step | Apple Silicon (MPS) | CPU only |
|------|:-------------------:|:--------:|
| 00 — fetch data | ~1 min | ~1 min |
| 01 — explore | ~5 sec | ~5 sec |
| 02 — baselines (RidgeCV) | ~5–10 min | ~10–20 min |
| 03 — PSO fly | ~1 min | ~5 min |
| 03 — PSO mouse | ~5–7 min | ~30 min |
| 03 — PSO rat | ~3–4 min | ~15 min |
| 03 — PSO macaque | ~25–35 min | ~2–3 hr |
| 04–05 — analysis | ~3 min | ~3 min |
| **Total** | **~45–65 min** | **~3–5 hr** |

---

## 🧪 Experimental Design

### Echo State Network (ESN)

Each connectome is used directly as the reservoir weight matrix.
A sparse random input weight matrix `w_in` projects a scalar signal into the network.
A Ridge regression readout maps the reservoir states to targets.

| Parameter | Value | Source |
|-----------|------:|--------|
| Spectral radius ρ | **0.97** | Dambre et al. 2012 — MC ↑ as ρ → 1 |
| Washout steps | 200 | Standard ESN practice |
| Signal length | 2 000 steps | — |
| Max lag (MC) | 50 | Extended from typical 20 |
| Train / test | 70% / 30% | — |
| Readout (baselines) | RidgeCV (α ∈ {0.01, 0.1, 1, 10, 100}, 5-fold) | Prevents overfitting |
| Readout (PSO inner loop) | Ridge (α = 1.0) | CV too slow in 1000-eval loop |

### Benchmark Tasks

**Memory Capacity (MC)**
$$MC = \sum_{k=1}^{50} R^2\bigl(u[t],\; \hat{y}[t-k]\bigr)$$
Input: i.i.d. Uniform[−1, 1]. Target: delayed recall up to 50 lags.
Higher is better.

**Lorenz Prediction**
One-step-ahead prediction of the Lorenz x-coordinate (σ=10, ρ=28, β=8/3).
Scored with NRMSE — lower is better.

### PSO Configuration

| Parameter | Value | Source |
|-----------|------:|--------|
| Cognitive c1 | 2.0 | Kennedy & Eberhart 1995 |
| Social c2 | 2.0 | Kennedy & Eberhart 1995 |
| Inertia w | 0.7 | Kennedy & Eberhart 1995 |
| Particles | 20 | — |
| Iterations | **50** | 1 000 evaluations/run |
| Weight bound | [0, bio\_max × 3] | Biologically plausible |
| PSO runs | 5 per condition | Gives paired distributions |

**Three conditions per species:**

| Condition | Initialisation | Purpose |
|-----------|---------------|---------|
| 🔵 **Biological** | Fixed bio weights | Reference baseline |
| 🟢 **PSO-A** | Gaussian(bio, 0.3·σ\_bio) | Fine-tune biology |
| 🟠 **PSO-B** | Uniform(0, max\_w) | Optimise from scratch |
| ⚫ **Random** | Uniform(0, max\_w), no PSO | Null control |

### Fair Evaluation (Anti-Selection-Bias)

A key design fix: PSO optimises weights on **signal A** (`seed = run`).
All reported scores — including the biological baseline — are evaluated on a
separate **held-out signal B** (`seed = run + 100`). This prevents PSO from
being compared against biology on the exact signal it was maximised for.

```
run 0:  PSO optimises on seed=0  →  all scores reported on seed=100
run 1:  PSO optimises on seed=1  →  all scores reported on seed=101
...
```

PSO particle initialisation uses a third independent seed (`run × 997 + 31`)
to prevent confounding between swarm diversity and signal variance.

### Statistical Analysis

- **Paired t-test** (df = 4): PSO-A vs Bio, PSO-B vs Bio, Random vs Bio
- **Cohen's d** reported alongside every test — essential at df = 4 where
  statistical power is limited (threshold: d ≈ 0.2 small, 0.5 medium, 0.8 large)
- Significance: `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05 · `ns` not significant

### Graph Topology

Metrics computed on **weighted directed** graphs (Fagiolo 2007):

| Metric | Method | Why not binary? |
|--------|--------|----------------|
| Clustering coefficient | `nx.average_clustering(DiGraph, weight=...)` | Weighted differs across conditions |
| Path length | Dijkstra with distance = 1/weight | Binary path = identical (topology unchanged) |
| Small-world σ | γ/λ, directed formula: mean\_k = edges/nodes | No factor-2 for directed |

---

## 📂 Outputs

| File | Description |
|------|-------------|
| `data/baseline_results.csv` | MC + Lorenz per run per species (bio only) |
| `data/pso_results.csv` | Held-out MC for all 4 conditions × all runs |
| `data/stats_summary.csv` | t-stat, p-value, Cohen's d, Δ% per comparison |
| `data/graph_metrics.csv` | Weighted directed clustering / path / σ |
| `data/input_sensitivity.csv` | Baseline MC across 4 input-node seeds |
| `data/summary.txt` | 📋 **LLM-ready full report text** — paste into GPT/Claude |
| `images/01_matrix_heatmaps.png` | Connectome weight matrices (log-scale for fly) |
| `images/02_degree_distributions.png` | In/out-degree histograms |
| `images/03_mc_baselines.png` | Biological MC baseline (mean ± SD) |
| `images/04_lorenz_baselines.png` | Biological Lorenz NRMSE baseline |
| `images/05_pso_convergence.png` | All 5 PSO runs (light) + best (bold) |
| `images/06_mc_boxplots.png` | Box plots with significance brackets + Cohen's d |
| `images/07_graph_metrics.png` | Weighted directed graph metric comparison |

---

## 🎨 Colour Convention

| Condition | Colour | Hex |
|-----------|--------|-----|
| 🔵 Biological | Blue | `#2196F3` |
| 🟢 PSO from biology (PSO-A) | Green | `#4CAF50` |
| 🟠 PSO from random (PSO-B) | Orange | `#FF5722` |
| ⚫ Random null | Grey | `#9E9E9E` |

---

## 📦 Dependencies

| Package | Role |
|---------|------|
| `conn2res` | ESN implementation on connectomes |
| `netneurotools` | Real connectome dataset fetching |
| `pyswarms` | GlobalBestPSO optimiser |
| `networkx` | Graph topology metrics |
| `scikit-learn` | RidgeCV readout |
| `scipy` | Lorenz ODE · paired t-tests · sparse eigenvalues |
| `torch` *(optional)* | MPS/CUDA batched ESN via `torch.bmm` |
| `numpy / pandas / matplotlib` | Core numerics, data, figures |

---

## 📚 References

- Chiang et al. (2011). Three-dimensional reconstruction of brain-wide wiring networks in Drosophila. *Current Biology*.
- Rubinov et al. (2015). Wiring cost and topological participation of the mouse brain connectome. *PNAS*.
- Bota et al. (2015). Architecture of the cerebral cortical association connectome underlying cognition. *PNAS*.
- Modha & Singh (2010). Network architecture of the long-distance pathways in the macaque brain. *PNAS*.
- Kennedy & Eberhart (1995). Particle swarm optimization. *ICNN*.
- Dambre et al. (2012). Information processing capacity of dynamical systems. *Scientific Reports*.
- Lukoševičius & Jaeger (2009). Reservoir computing approaches to recurrent neural network training. *Computer Science Review*.
- Fagiolo (2007). Clustering in complex directed networks. *Physical Review E*.
- Verstraeten et al. (2010). An experimental unification of reservoir computing methods. *Neural Networks*.
