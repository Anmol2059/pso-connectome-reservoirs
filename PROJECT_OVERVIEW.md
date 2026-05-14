# 📋 Project Overview — Wild Reservoirs

Full experimental design, parameter choices, and scientific rationale.

---

## 🧠 Reservoir Computing Background

An **Echo State Network (ESN)** is a recurrent neural network with a fixed random reservoir.
Only the output (readout) weights are trained — making it fast and analytically tractable.

The key insight of this project: instead of a random reservoir, use a **real biological connectome**.
The brain has already been optimised by evolution — can we do better?

```
Input signal u(t)
      ↓
[W_in]  ←  sparse fixed input weights
      ↓
[Reservoir W]  ←  biological connectome (fixed topology, optimisable weights)
      |
   s(t) = tanh(s(t-1) · W + u(t) · W_in)
      ↓
[W_out]  ←  trained Ridge readout
      ↓
Output ŷ(t)
```

### ESN Hyperparameters

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| Spectral radius ρ | **0.97** | MC increases monotonically as ρ→1; 0.95–0.99 is standard for MC (Dambre et al. 2012, Verstraeten et al. 2010) |
| Washout | 200 steps | Eliminates transient initial states |
| Signal length | 2 000 steps | After washout |
| MC lags | 50 | Extended from typical 20 — captures medium-range memory |
| Train / test | 70% / 30% | — |

---

## 🐾 Connectome Details

### 🪱 *C. elegans* — Varshney et al. 2011
- **279 neurons**, 2 194 directed synaptic connections
- Chemical-synapse count weights (integer values)
- **Input nodes:** sensory neurons — mechanosensory (ALM, AVM, PLM, PVM) + chemosensory (ASE, ASH, ASI, AWA, AWB, AWC, AFD, …) — ~65 cells, biologically motivated
- Source: [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1001066)

### 🪰 *Drosophila* — Chiang et al. 2011
- **49 regions**, 1 950 connections, density 82.9%
- Continuous projection-strength weights
- **Input nodes:** olfactory network (`olf` label) — biologically motivated
- Source: [Current Biology](https://doi.org/10.1016/j.cub.2010.11.056)

### 🐭 Mouse cortex — Rubinov et al. 2015
- **112 regions**, 6 542 connections, density 52.6%
- Continuous axonal-tracing weights (anterograde)
- **Input nodes:** random 20% (seed=42)
- Source: [PNAS](https://doi.org/10.1073/pnas.1420315112)

### 🐀 Rat cortex — Bota et al. 2015
- **73 regions**, 1 923 connections, density 36.6%
- Continuous axonal-tracing weights
- **Input nodes:** random 20% (seed=42)
- Source: [PNAS](https://doi.org/10.1073/pnas.1504394112)

### 🐒 Macaque — binary — Modha & Singh 2010
- **242 regions**, 4 090 connections, density 7.0%
- **Binary** (0/1) — only presence/absence of long-distance pathways
- **Input nodes:** random 20% (seed=42)
- Source: [PNAS](https://doi.org/10.1073/pnas.1008054107)

### 🐒 Macaque — weighted (FLNe) — Markov et al. 2014
- **29 core cortical areas**, near-complete connectivity
- **Continuous FLNe weights** (Fraction of Labelled Neurons — axonal tracing)
- Same species as binary macaque → isolates topology vs. weight encoding
- Extracted from a 93×29 rectangular dataset; 29×29 square sub-matrix of injection sites that are also target areas
- **Input nodes:** random 20% (seed=42)
- Source: [Cerebral Cortex](https://doi.org/10.1093/cercor/bhs270)

### 🧑 Human cortex — Cammoun et al. 2012
- **83 parcels**, Lausanne atlas scale-033
- **DTI tractography** — diffusion MRI streamline density, group-averaged (~70 adults)
- Symmetric and undirected (unlike axonal-tracing species — a deliberate contrast)
- **Input nodes:** primary sensorimotor and visual parcels (precentral, postcentral, pericalcarine, transverse temporal)
- Source: [J. Neuroscience Methods](https://doi.org/10.1016/j.jneumeth.2011.09.031)

> ⚠️ Human data is non-invasive (MRI) vs axonal tracing in animals. This is scientifically
> accepted and used in many recent papers — but it is a different measurement modality.
> Results for humans should be interpreted accordingly.

---

## 🦾 Optimiser Details

### 🐦 PSO — Particle Swarm Optimisation

Each particle is a flattened weight matrix. Particles fly through solution space,
attracted to their own best-ever position (cognitive term) and the swarm's global best (social term).

```
v(t+1) = w·v(t) + c1·r1·(pbest - x) + c2·r2·(gbest - x)
x(t+1) = x(t) + v(t+1)
```

| Parameter | Value |
|-----------|------:|
| Inertia w | 0.7 |
| Cognitive c1 | 2.0 |
| Social c2 | 2.0 |
| Particles | 20 |
| Iterations | 50 |

### 🧬 DE — Differential Evolution (DE/rand/1/bin)

A population-based search that creates trial vectors by adding scaled differences between
random members, then accepts them if they improve on the target.

```
mutant = x_r1 + F · (x_r2 - x_r3)          # mutation
trial[j] = mutant[j] if rand < CR else x[j]  # binomial crossover
```

| Parameter | Value |
|-----------|------:|
| Scale factor F | 0.8 |
| Crossover rate CR | 0.9 |
| Population | 20 |
| Generations | 50 |

### 🐺 GWO — Grey Wolf Optimiser

The three best solutions at each iteration are the alpha (α), beta (β), and delta (δ) wolves.
Remaining wolves update their position by encircling the estimated prey position from all three leaders.

```
D_α = |C1·X_α - X|,   X1 = X_α - A1·D_α
D_β = |C2·X_β - X|,   X2 = X_β - A2·D_β
D_δ = |C3·X_δ - X|,   X3 = X_δ - A3·D_δ
X(t+1) = (X1 + X2 + X3) / 3
```

| Parameter | Value |
|-----------|------:|
| a (linearly decreases) | 2 → 0 |
| Pack size | 20 |
| Iterations | 50 |

**All three algorithms share:** 20 population × 50 iterations = **1 000 evaluations per run**.

---

## 🧪 Tasks

### 💾 Memory Capacity (MC)

$$MC = \sum_{k=1}^{50} R^2\!\bigl(u[t-k],\; \hat{u}[t-k]\bigr)$$

- Input: i.i.d. Uniform[−1, 1] signal
- Readout trained to reconstruct delayed input at lags k = 1 … 50
- Theoretical maximum MC = N (number of nodes)
- Reference: [Dambre et al. 2012](https://doi.org/10.1038/srep00514)

### 🌀 Lorenz Attractor Prediction

$$\dot{x} = \sigma(y-x), \quad \dot{y} = x(\rho-z)-y, \quad \dot{z} = xy - \beta z$$

Parameters: σ=10, ρ=28, β=8/3, dt=0.02. Reservoir driven by x-coordinate.
Readout trained for one-step-ahead prediction. Scored with NRMSE (lower = better).

---

## 🎯 Experimental Conditions

| Condition | Init | Optimised? | Purpose |
|-----------|------|:----------:|---------|
| **Biological** | Fixed bio weights | ❌ | Nature's baseline |
| **Algo-A** (PSO-A / DE-A / GWO-A) | Gaussian(bio, 0.3·σ_bio) | ✅ | Fine-tune biology |
| **Algo-B** (PSO-B / DE-B / GWO-B) | Uniform(0, max_w) | ✅ | Optimise from scratch |
| **Random** | Uniform(0, max_w) | ❌ | Null control |

### Anti-Selection-Bias (Held-Out Evaluation)

This is the most important design decision. Without it, results are meaningless.

```
❌ WRONG:  PSO optimises on signal S → report PSO score on S
           Compare Bio score on S → PSO wins trivially (it was maximised on S!)

✅ CORRECT: PSO optimises on signal A (seed = run)
            All scores (Bio, PSO, Random) reported on signal B (seed = run + 100)
```

```
Run 0:  optimise → seed 0    |  report → seed 100
Run 1:  optimise → seed 1    |  report → seed 101
Run 2:  optimise → seed 2    |  report → seed 102
Run 3:  optimise → seed 3    |  report → seed 103
Run 4:  optimise → seed 4    |  report → seed 104
```

---

## 📊 Statistical Analysis

- **5 independent runs** per (species × algorithm × task × condition)
- **Paired t-test** (df = 4): optimised condition vs biological, paired by run index
- **Cohen's d** reported alongside every p-value (essential at df=4 where power is limited)
  - d < 0.2: negligible · d ≈ 0.5: medium · d > 0.8: large
- Significance: `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `ns` not significant

---

## ⚙️ GPU Acceleration

All P=20 candidate networks are evaluated **simultaneously** in one batched `torch.bmm` call:

```python
# Shape: (P, T, N) × (P, N, N) → (P, T, N)  — all particles in one GPU pass
states = torch.bmm(states, W_batch) + input_proj
```

This gives ~10–30× speedup vs sequential CPU evaluation for medium/large connectomes.
Falls back to CPU automatically if no GPU is available.

---

## 🗺️ Graph Topology Metrics

After optimisation, we compare the weighted directed graph structure (Fagiolo 2007):

| Metric | Method |
|--------|--------|
| Clustering coefficient | `nx.average_clustering(DiGraph, weight=...)` |
| Average path length | Dijkstra with distance = 1/weight |
| Small-world coefficient σ | γ / λ, directed formula |

Binary path length is identical across conditions (topology unchanged) — weights are required
to detect changes. Reference: [Fagiolo 2007](https://doi.org/10.1103/PhysRevE.76.026107)

---

## 📦 Dependencies

| Package | Role |
|---------|------|
| [`netneurotools`](https://github.com/netneurolab/netneurotools) | Download all 7 real connectomes |
| [`conn2res`](https://github.com/netneurolab/conn2res) | ESN implementation on connectomes |
| `torch` | GPU-batched ESN via `torch.bmm` |
| `scikit-learn` | RidgeCV readout |
| `scipy` | Lorenz ODE · sparse eigenvalues · t-tests |
| `networkx` | Weighted directed graph metrics |
| `numpy / pandas / matplotlib` | Numerics, data, figures |

---

## 📚 Full Reference List

**Connectome data:**
- Varshney et al. (2011). Structural properties of the *C. elegans* neuronal network. *PLOS Comp. Biol.* [doi](https://doi.org/10.1371/journal.pcbi.1001066)
- Chiang et al. (2011). Three-dimensional reconstruction of brain-wide wiring networks in *Drosophila*. *Current Biology.* [doi](https://doi.org/10.1016/j.cub.2010.11.056)
- Rubinov et al. (2015). Wiring cost and topological participation of the mouse brain connectome. *PNAS.* [doi](https://doi.org/10.1073/pnas.1420315112)
- Bota et al. (2015). Architecture of the cerebral cortical association connectome underlying cognition. *PNAS.* [doi](https://doi.org/10.1073/pnas.1504394112)
- Modha & Singh (2010). Network architecture of the long-distance pathways in the macaque brain. *PNAS.* [doi](https://doi.org/10.1073/pnas.1008054107)
- Markov et al. (2014). A weighted and directed interareal connectivity matrix for macaque cerebral cortex. *Cerebral Cortex.* [doi](https://doi.org/10.1093/cercor/bhs270)
- Cammoun et al. (2012). Mapping the human connectome at multiple scales. *J. Neurosci. Methods.* [doi](https://doi.org/10.1016/j.jneumeth.2011.09.031)
- Markello et al. (2022). netneurotools: A Python toolbox for network neuroscience. [GitHub](https://github.com/netneurolab/netneurotools)

**Optimisers:**
- Kennedy & Eberhart (1995). Particle swarm optimization. *IEEE ICNN.* [doi](https://doi.org/10.1109/ICNN.1995.488968)
- Storn & Price (1997). Differential evolution — a simple and efficient heuristic for global optimization. *J. Global Optimization.* [doi](https://doi.org/10.1023/A:1008202821328)
- Mirjalili et al. (2014). Grey wolf optimizer. *Advances in Engineering Software.* [doi](https://doi.org/10.1016/j.advengsoft.2013.12.007)

**Reservoir computing:**
- Lukoševičius & Jaeger (2009). Reservoir computing approaches to recurrent neural network training. *Computer Science Review.* [doi](https://doi.org/10.1016/j.cosrev.2009.03.005)
- Dambre et al. (2012). Information processing capacity of dynamical systems. *Scientific Reports.* [doi](https://doi.org/10.1038/srep00514)
- Verstraeten et al. (2010). An experimental unification of reservoir computing methods. *Neural Networks.* [doi](https://doi.org/10.1016/j.neunet.2007.04.003)
- Fagiolo (2007). Clustering in complex directed networks. *Physical Review E.* [doi](https://doi.org/10.1103/PhysRevE.76.026107)
