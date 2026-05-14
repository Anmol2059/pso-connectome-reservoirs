# Wild Reservoirs: Bio-Inspired Optimisation of Neural Connectomes from Worm to Human

> **Can algorithms born from animal behaviour rediscover what evolution wired into animal brains?**
> We take real biological connectomes — from the 302-neuron worm to the human cortex — plug them
> directly into reservoir computers, then unleash swarm and evolutionary optimisers to see whether
> they can outperform 500 million years of natural selection.

**Course:** Bio-Inspired Learning &nbsp;|&nbsp; **Institution:** ETSI Telecomunicación, UPM Madrid &nbsp;|&nbsp; **Format:** IEEE

---

## The Experiment in One Sentence

Seven real brain connectomes. Three bio-inspired optimisers. Two computational tasks.
Each optimiser tries to rewire the weights of a biological network; we ask whether it beats nature.

---

## Biological Connectomes

All connectomes are downloaded automatically via `netneurotools.datasets.fetch_famous_gmat`.
No synthetic data. Every network is a real measurement from peer-reviewed neuroscience.

| Animal | Nodes | Edges | Density | Weight type | Input nodes | Source |
|--------|------:|------:|--------:|-------------|-------------|--------|
| *C. elegans* (roundworm) | 279 | 2 194 | 2.8% | Continuous (synapse counts) | Sensory neurons (mechanosensory + chemosensory, ~65 cells) | Varshney et al. 2011 |
| *Drosophila* (fruit fly) | 49 | 1 950 | 82.9% | Continuous (projection strength) | Olfactory network | Chiang et al. 2011 |
| Mouse cortex | 112 | 6 542 | 52.6% | Continuous (axonal tracing) | Random 20% (seed=42) | Rubinov et al. 2015 |
| Rat cortex | 73 | 1 923 | 36.6% | Continuous (axonal tracing) | Random 20% (seed=42) | Bota et al. 2015 |
| Macaque — binary | 242 | 4 090 | 7.0% | **Binary** (0/1 connectivity) | Random 20% (seed=42) | Modha & Singh 2010 |
| Macaque — weighted | 29 | 841 | 99.9% | Continuous (FLNe tracing) | Random 20% (seed=42) | Markov et al. 2014 |
| Human cortex | 83 | ~3 500 | ~52% | Continuous (DTI tractography) | Sensorimotor/visual parcels | Cammoun et al. 2012 |

**Why two macaque networks?** Macaque-binary and macaque-weighted share the same anatomical
structure (same species, same cortical area set) but differ only in weight encoding.
This isolates whether optimisers respond to *topology* or *weight magnitudes*.

**Human data note:** Human structural connectivity is measured non-invasively by diffusion
MRI tractography (Lausanne scale-033, group-averaged ~70 adults). It is symmetric and
undirected, unlike the axonal-tracing networks from other species — a deliberately chosen
contrast that broadens the phylogenetic scope from worm to human.

---

## Bio-Inspired Optimisers

Three algorithms, each inspired by a different kind of collective animal intelligence:

### Particle Swarm Optimisation (PSO)
Mimics bird flocking and fish schooling. Each particle (a candidate weight matrix) moves
through the solution space attracted to its personal best and the swarm's global best.

| Parameter | Value |
|-----------|------:|
| Inertia w | 0.7 |
| Cognitive c1 | 2.0 |
| Social c2 | 2.0 |
| Source | Kennedy & Eberhart, ICNN 1995 |

### Differential Evolution (DE/rand/1/bin)
Mimics genetic mutation and recombination in a population of solutions.
A trial vector is created by adding scaled differences between random population members;
binomial crossover accepts it if it improves on the target.

| Parameter | Value |
|-----------|------:|
| Scale factor F | 0.8 |
| Crossover rate CR | 0.9 |
| Source | Storn & Price, *J. Global Optimization* 1997 |

### Grey Wolf Optimiser (GWO)
Mimics the hunting hierarchy of grey wolf packs. The three best solutions are named
alpha, beta, and delta wolves; the remaining population updates its position by
encircling the hunt position estimated from all three leaders.

| Parameter | Value |
|-----------|------:|
| a decreases | 2 → 0 linearly |
| Source | Mirjalili et al., *Advances in Engineering Software* 2014 |

**All three share the same budget:** 20 particles × 50 iterations = 1 000 objective evaluations per run.

---

## Computational Tasks

### Memory Capacity (MC)
Reservoir is driven by an i.i.d. Uniform[−1, 1] signal. A Ridge readout is trained to
reconstruct the input delayed by k steps, for k = 1 … 50.

$$MC = \sum_{k=1}^{50} R^2\!\bigl(u[t-k],\; \hat{u}[t-k]\bigr)$$

Higher MC = longer memory span. Theoretical maximum for N nodes is N.

### Lorenz Attractor Prediction
Reservoir is driven by the x-coordinate of a Lorenz trajectory (σ=10, ρ=28, β=8/3, dt=0.02).
A Ridge readout predicts the *next* x value (one-step-ahead).

$$\text{NRMSE} = \frac{\sqrt{\langle (y - \hat{y})^2 \rangle}}{y_{\max} - y_{\min}}$$

Lower NRMSE = better chaotic forecasting.

---

## Experimental Design

### Echo State Network Setup

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| Spectral radius ρ | **0.97** | MC increases monotonically as ρ → 1 (Dambre et al. 2012) |
| Washout | 200 steps | Eliminates transient initial states |
| Signal length | 2 000 steps | After washout |
| MC lags | 50 | Captures medium-range memory |
| Train / test split | 70% / 30% | — |
| Baseline readout | RidgeCV, α ∈ {0.01, 0.1, 1, 10, 100}, 5-fold | Cross-validated, prevents overfitting |
| Optimiser readout | Ridge, α = 1.0 (fixed) | CV too slow inside 1 000-eval loop |

### Four Experimental Conditions

| Condition | Initialisation | Purpose |
|-----------|---------------|---------|
| **Biological** | Fixed bio weights (no optimisation) | Nature's solution |
| **PSO-A / DE-A / GWO-A** | Gaussian(bio, 0.3·σ_bio) — close to biology | Fine-tune biological structure |
| **PSO-B / DE-B / GWO-B** | Uniform(0, max_w) — random start | Optimise from scratch |
| **Random** | Uniform(0, max_w), not optimised | Null control |

### Anti-Selection-Bias (Held-Out Evaluation)

A critical design decision: optimisers work on **signal A** (`seed = run`).
All reported scores — biological, optimised, random — are evaluated on a separate
**held-out signal B** (`seed = run + 100`).

```
Run 0:  optimise on seed=0   →  report scores on seed=100
Run 1:  optimise on seed=1   →  report scores on seed=101
Run 2:  optimise on seed=2   →  report scores on seed=102
...
```

Without this, PSO would be compared against biology on the exact signal it maximised — an
unfair advantage that would inflate apparent optimiser gains.

### GPU Acceleration

All P=20 candidate networks are evaluated **simultaneously** via a single batched
`torch.bmm` call on the GPU (NVIDIA RTX 6000 Ada, CUDA 12.8). The ESN simulation loop
runs in Python but matrix multiplications are offloaded to GPU — giving ~10–30× speedup
over sequential CPU evaluation for medium/large connectomes.

### Statistical Analysis

- **5 independent runs** per (species, algorithm, task, condition)
- **Paired t-test** (df = 4): optimised vs biological, within the same run index
- **Cohen's d** reported alongside every p-value (essential at df = 4 where power is limited)
- Significance: `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05 · `ns` not significant

---

## Quick Start

```bash
git clone https://github.com/Anmol2059/pso-connectome-reservoirs.git
cd pso-connectome-reservoirs

# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Install conn2res (ESN library, no-deps install)
git clone https://github.com/netneurolab/conn2res.git
cd conn2res && pip install --no-deps . && cd ..

# 3. Run the full pipeline  (~60-90 min on GPU, ~4-6 hr on CPU)
cd project
bash run.sh
```

> **GPU (NVIDIA):** PyTorch auto-detects CUDA. All 20-particle batches run as a single `torch.bmm`.
> **Apple Silicon:** MPS backend is used automatically — significant speedup vs CPU.
> **CPU-only:** Still works; just slower on large connectomes (macaque, C. elegans).

---

## Pipeline

```
bash run.sh
│
├── 00  fetch_real_connectomes.py   Download & preprocess 7 biological connectomes
│                                   → data/{species}/conn.npy  (weight matrix)
│                                   → data/{species}/labels.npy (input node flags)
│
├── 01  step01_explore.py           Visualise weight matrices & degree distributions
│                                   → images/01_matrix_heatmaps.png
│                                   → images/02_degree_distributions.png
│
├── 02  step02_baselines.py         MC + Lorenz baselines for all 7 species
│                                   5 runs × RidgeCV readout × held-out eval
│                                   → data/baseline_results.csv
│                                   → images/03_mc_baselines.png
│                                   → images/04_lorenz_baselines.png
│
├── 03  step03_pso.py               PSO + DE + GWO optimisation
│                                   3 algorithms × 2 tasks × 7 species × 5 runs
│                                   GPU-batched, held-out evaluation
│                                   → data/opt_results.csv
│                                   → images/05_convergence.png
│
├── 04  step04_analysis.py          Paired t-tests + Cohen's d
│                                   Weighted directed graph topology metrics
│                                   Input-node sensitivity analysis
│                                   → data/stats_summary.csv
│                                   → data/graph_metrics.csv
│                                   → data/input_sensitivity.csv
│                                   → images/06_mc_boxplots.png
│                                   → images/07_lorenz_boxplots.png
│                                   → images/08_graph_metrics.png
│
└── 05  step05_summary.py           Compile full results → data/summary.txt
```

Each step logs to `logs/step0X.log` and can be re-run independently.

---

## Expected Runtime

| Step | NVIDIA GPU | CPU only |
|------|:----------:|:--------:|
| 00 — fetch connectomes | ~1 min | ~1 min |
| 01 — explore & plot | ~5 sec | ~5 sec |
| 02 — baselines (7 species) | ~5 min | ~20 min |
| 03 — optimise (PSO+DE+GWO, 7 species) | ~60–90 min | ~4–6 hr |
| 04–05 — analysis + summary | ~3 min | ~3 min |
| **Total** | **~70–100 min** | **~5–7 hr** |

---

## Outputs

| File | Description |
|------|-------------|
| `data/baseline_results.csv` | MC + Lorenz per run, 7 species, biological weights |
| `data/opt_results.csv` | Held-out scores, all conditions × algorithms × species |
| `data/stats_summary.csv` | t-stat, p-value, Cohen's d, Δ% for every comparison |
| `data/graph_metrics.csv` | Weighted directed clustering / path / small-world σ |
| `data/input_sensitivity.csv` | MC baseline across 4 input-node seeds |
| `data/summary.txt` | Full results report (human + LLM readable) |

---

## Acknowledgements

This work builds on open neuroscience datasets and open-source tools.
We thank the teams behind:

- **Varshney et al. 2011** — *C. elegans* connectome (279 chemical-synapse neurons). [PLOS Comp. Biol.](https://doi.org/10.1371/journal.pcbi.1001066)
- **Chiang et al. 2011** — *Drosophila* whole-brain wiring (49 regions). [Current Biology](https://doi.org/10.1016/j.cub.2010.11.056)
- **Rubinov et al. 2015** — Mouse cortex connectome (112 regions). [PNAS](https://doi.org/10.1073/pnas.1420315112)
- **Bota et al. 2015** — Rat cortical connectome (73 regions). [PNAS](https://doi.org/10.1073/pnas.1504394112)
- **Modha & Singh 2010** — Macaque long-distance pathways, binary (242 regions). [PNAS](https://doi.org/10.1073/pnas.1008054107)
- **Markov et al. 2014** — Macaque cortex, FLNe continuous weights (29 core areas). [Cerebral Cortex](https://doi.org/10.1093/cercor/bhs270)
- **Cammoun et al. 2012** — Human structural MRI, Lausanne atlas scale-033 (83 parcels). [J. Neurosci. Methods](https://doi.org/10.1016/j.jneumeth.2011.09.031)
- **netneurotools** (Markello et al. 2022) — Python toolbox for accessing all the above datasets. [GitHub](https://github.com/netneurolab/netneurotools)
- **conn2res** — Reservoir computing on connectomes. [GitHub](https://github.com/netneurolab/conn2res)

### Algorithm references

- Kennedy & Eberhart (1995) — Particle swarm optimization. *IEEE ICNN*. [DOI](https://doi.org/10.1109/ICNN.1995.488968)
- Storn & Price (1997) — Differential evolution. *J. Global Optimization*. [DOI](https://doi.org/10.1023/A:1008202821328)
- Mirjalili et al. (2014) — Grey wolf optimizer. *Advances in Engineering Software*. [DOI](https://doi.org/10.1016/j.advengsoft.2013.12.007)

### Reservoir computing references

- Dambre et al. (2012) — Information processing capacity of dynamical systems. *Scientific Reports*. [DOI](https://doi.org/10.1038/srep00514)
- Lukoševičius & Jaeger (2009) — Reservoir computing approaches to RNN training. *Computer Science Review*. [DOI](https://doi.org/10.1016/j.cosrev.2009.03.005)
- Verstraeten et al. (2010) — An experimental unification of reservoir computing methods. *Neural Networks*. [DOI](https://doi.org/10.1016/j.neunet.2007.04.003)
- Fagiolo (2007) — Clustering in complex directed networks. *Physical Review E*. [DOI](https://doi.org/10.1103/PhysRevE.76.026107)

---

## Authors

Anmol Guragain · Savvas Kakalis · Juan Ignacio Godino Llorente

ETSI de Telecomunicación, Universidad Politécnica de Madrid, Madrid, Spain
