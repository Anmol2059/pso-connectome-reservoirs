# Project Overview — Connectome Reservoir Optimisation

Full experimental design, parameter choices, and scientific rationale behind the paper
*"The Whale That Outswam Evolution: Swarm Intelligence Maximises Memory in Connectome Reservoirs"*.

---

## Reservoir Computing Background

An **Echo State Network (ESN)** is a recurrent neural network with a fixed recurrent reservoir. Only the output (readout) weights are trained, making it fast and analytically tractable.

The central idea of this project: instead of a random reservoir, use a **real biological connectome**. Evolution has already optimised these networks for efficient temporal computation — but can gradient-free swarm optimisers do even better, starting from biology?

```
Input signal u(t)
      ↓
[W_in]  ←  sparse fixed input weights
      ↓
[Reservoir W]  ←  biological connectome (fixed topology, optimised weights)
      |
   x(t) = tanh(W·x(t−1) + W_in·u(t))
      ↓
[W_out]  ←  trained Ridge readout
      ↓
Output ŷ(t)
```

### ESN Hyperparameters

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| Spectral radius ρ | 0.97 | MC increases monotonically as ρ→1; Dambre et al. 2012 |
| Washout | 200 steps | Eliminates transient initial states |
| Signal length | 2,000 steps | Standard RC evaluation length |
| MC lags | 50 | Covers memory range of smaller reservoirs |
| Train / test split | 70% / 30% | Standard |

---

## Six Biological Connectomes

### *C. elegans* — Varshney et al. 2011
- **279 neurons**, 2,194 directed synaptic connections, density 2.8%
- Chemical-synapse count weights (integer values)
- Input nodes: random 20% (seed=42)
- Source: [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1001066)

### *Drosophila* — Chiang et al. 2011
- **49 regions**, 1,950 connections, density 82.9%
- Continuous projection-strength weights (confocal fluorescence)
- Input nodes: olfactory network (`olf` label) — biologically motivated
- Source: [Current Biology](https://doi.org/10.1016/j.cub.2010.11.056)

### Mouse cortex — Rubinov et al. 2015
- **112 regions**, 6,542 connections, density 52.6%
- Continuous axonal-tracing weights (anterograde injection density)
- Input nodes: random 20% (seed=42)
- Source: [PNAS](https://doi.org/10.1073/pnas.1420315112)

### Rat cortex — Bota et al. 2015
- **73 regions**, 1,923 connections, density 36.6%
- Continuous connectivity-strength weights (axonal tracing)
- Input nodes: random 20% (seed=42)
- Source: [PNAS](https://doi.org/10.1073/pnas.1504394112)

### Macaque — Markov et al. 2014 (weighted FLNe)
- **29 core cortical areas**, 590 connections, density 72.7%
- Continuous FLNe weights (Fraction of Labelled Neurons — retrograde tracing)
- The only connectome with near-complete, continuous interareal weights
- Input nodes: random 20% (seed=42)
- Source: [Cerebral Cortex](https://doi.org/10.1093/cercor/bhs270)

> Note: A binary macaque connectome (Modha & Singh 2010) was considered but excluded — discrete 0/1 weights provide no continuous optimisation landscape for gradient-free search.

### Human cortex — Cammoun et al. 2012
- **83 parcels**, 2,134 connections, density 31.4% (Lausanne atlas scale-033)
- DTI tractography — diffusion MRI streamline density, group-averaged (~70 adults)
- Symmetric and undirected (unlike axonal-tracing datasets)
- Input nodes: random 20% (seed=42)
- Source: [J. Neuroscience Methods](https://doi.org/10.1016/j.jneumeth.2011.09.031)

> Human data uses non-invasive MRI rather than histological tracing. Results are valid and consistent with prior literature but represent an indirect proxy for true axonal connectivity.

---

## Four Bio-Inspired Optimisers

All four algorithms are gradient-free, population-based, and operate on the non-zero edge weights while the sparsity pattern remains frozen. All share an identical budget: **20 particles × 50 iterations = 1,000 fitness evaluations per run**.

### PSO — Particle Swarm Optimisation (Kennedy & Eberhart 1995)

```
v(t+1) = w·v(t) + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x(t+1) = x(t) + v(t+1)
```

| Parameter | Value |
|-----------|------:|
| Inertia w | 0.7 |
| Cognitive c1 | 2.0 |
| Social c2 | 2.0 |

### DE — Differential Evolution, DE/rand/1/bin (Storn & Price 1997)

```
mutant = x_r1 + F · (x_r2 − x_r3)
trial[j] = mutant[j] if rand < CR else x[j]
```

| Parameter | Value |
|-----------|------:|
| Scale factor F | 0.8 |
| Crossover rate CR | 0.9 |

### GWO — Grey Wolf Optimiser (Mirjalili et al. 2014)

The three best solutions (α, β, δ wolves) guide the swarm via position averaging with a linearly decaying coefficient a: 2→0.

```
X(t+1) = (X_α − A1·D_α + X_β − A2·D_β + X_δ − A3·D_δ) / 3
```

### WOA — Whale Optimisation Algorithm (Mirjalili & Lewis 2016)

Models humpback whale bubble-net feeding. Switches between three mechanisms based on a random probability p:
- **p < 0.5, |A| < 1:** Shrinking encircling (exploitation)
- **p < 0.5, |A| ≥ 1:** Random search (exploration)
- **p ≥ 0.5:** Logarithmic spiral attack (bubble-net)

```
X(t+1) = X*(t) − A·D    [encircling]
X(t+1) = D'·e^(b·l)·cos(2πl) + X*(t)    [spiral, b=1.0]
```

Linearly decaying a: 2→0 across iterations.

---

## Four Benchmark Tasks

### Memory Capacity (MC)

$$MC = \sum_{k=1}^{50} r^2\bigl(u(t-k),\; \hat{u}(t-k)\bigr)$$

- Input: i.i.d. Uniform[−1, 1]
- Readout trained to reconstruct delayed versions of input at lags k = 1…50
- Theoretical maximum MC = N (reservoir size)
- Higher is better

### Lorenz Attractor Prediction

$$\dot{x} = \sigma(y-x),\quad \dot{y} = x(\rho-z)-y,\quad \dot{z} = xy - \beta z$$

Parameters: σ=10, ρ=28, β=8/3, dt=0.02. Reservoir driven by x-coordinate; readout performs one-step-ahead prediction. NRMSE — lower is better.

### NARMA-10

$$y(t) = 0.3y(t{-}1) + 0.05y(t{-}1)\sum_{i=1}^{10}y(t{-}i) + 1.5\,u(t{-}10)\,u(t{-}1) + 0.1$$

Input: u(t) ~ Uniform[0, 0.5]. Requires both nonlinear computation and ten-step memory simultaneously. NRMSE — lower is better.

### Mackey–Glass Prediction

$$\dot{x}(t) = \frac{0.2\,x(t-\tau)}{1 + x(t-\tau)^{10}} - 0.1\,x(t)$$

τ=17 (chaotic regime), integrated at dt=0.1. One-step-ahead prediction of the scalar time series. NRMSE — lower is better.

---

## Experimental Conditions

| Condition | Initialisation | Optimised | Purpose |
|-----------|---------------|:---------:|---------|
| **Bio** | Fixed biological weights | No | Evolution's baseline |
| **PSO-A / DE-A / GWO-A / WOA-A** | Gaussian(w_bio, 0.3·σ_bio), clipped to [0, 3·w_max] | Yes | Fine-tune from biology |
| **PSO-B** | Uniform random (MC task only) | Yes | Negative control — no bio initialisation |

### Held-Out Evaluation Protocol

Every run uses a separate random seed for optimisation and evaluation to prevent selection bias:

```
Run r:  optimise on seed r  →  report score on seed r+100
```

This means the optimiser never sees the signal it is evaluated on, preventing artificially inflated scores.

---

## Statistical Analysis

- **10 independent runs** per (species × algorithm × task × condition)
- **Paired t-test** (df=9): optimised condition vs biological baseline, paired by run index
- All WOA-A vs Bio comparisons significant at p < 0.001 across all 24 species–task combinations
- Pearson correlation: reservoir size N vs WOA MC improvement (r=0.97, p=0.001)
- Pearson correlation: synaptic density vs WOA MC improvement (r=−0.83, p=0.042)

---

## GPU Acceleration

All P=20 candidate networks are evaluated simultaneously in one batched PyTorch operation:

```python
# Shape: (P, T, N) — all particles in one GPU pass
states = torch.tanh(torch.bmm(states, W_batch) + input_proj)
```

This gives ~10–20× speedup vs sequential evaluation. Falls back to CPU automatically if no GPU is available.

---

## Dependencies

| Package | Role |
|---------|------|
| [`netneurotools`](https://github.com/netneurolab/netneurotools) | Download all 6 real connectomes |
| [`conn2res`](https://github.com/netneurolab/conn2res) | ESN implementation on connectomes |
| `torch` | GPU-batched ESN evaluation |
| `scikit-learn` | RidgeCV readout |
| `scipy` | ODE integration, paired t-tests, Pearson correlation |
| `networkx` | Graph topology metrics |
| `numpy / pandas / matplotlib` | Numerics, data, figures |

---

## References

**Connectome data:**
- Varshney et al. (2011). Structural properties of the *C. elegans* neuronal network. *PLOS Comp. Biol.* [doi](https://doi.org/10.1371/journal.pcbi.1001066)
- Chiang et al. (2011). Three-dimensional reconstruction of brain-wide wiring networks in *Drosophila*. *Current Biology.* [doi](https://doi.org/10.1016/j.cub.2010.11.056)
- Rubinov et al. (2015). Wiring cost and topological participation of the mouse brain connectome. *PNAS.* [doi](https://doi.org/10.1073/pnas.1420315112)
- Bota et al. (2015). Architecture of the cerebral cortical association connectome. *PNAS.* [doi](https://doi.org/10.1073/pnas.1504394112)
- Markov et al. (2014). A weighted and directed interareal connectivity matrix for macaque cerebral cortex. *Cerebral Cortex.* [doi](https://doi.org/10.1093/cercor/bhs270)
- Cammoun et al. (2012). Mapping the human connectome at multiple scales. *J. Neurosci. Methods.* [doi](https://doi.org/10.1016/j.jneumeth.2011.09.031)
- Markello et al. (2022). netneurotools. [GitHub](https://github.com/netneurolab/netneurotools)

**Optimisers:**
- Kennedy & Eberhart (1995). Particle swarm optimization. *IEEE ICNN.* [doi](https://doi.org/10.1109/ICNN.1995.488968)
- Storn & Price (1997). Differential evolution. *J. Global Optimization.* [doi](https://doi.org/10.1023/A:1008202821328)
- Mirjalili et al. (2014). Grey wolf optimizer. *Advances in Engineering Software.* [doi](https://doi.org/10.1016/j.advengsoft.2013.12.007)
- Mirjalili & Lewis (2016). The whale optimization algorithm. *Advances in Engineering Software.* [doi](https://doi.org/10.1016/j.advengsoft.2016.01.008)

**Reservoir computing:**
- Jaeger (2001). The echo state approach to analysing and training recurrent neural networks. GMD Report 148.
- Dambre et al. (2012). Information processing capacity of dynamical systems. *Scientific Reports.* [doi](https://doi.org/10.1038/srep00514)
- Suárez et al. (2024). Connectome-based reservoir computing with the conn2res toolbox. *Nature Communications.* [doi](https://doi.org/10.1038/s41467-024-44900-4)
