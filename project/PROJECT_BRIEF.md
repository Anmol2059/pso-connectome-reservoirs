# Bio-Inspired Learning — Project Brief
## PSO-Optimised Connectome Reservoirs: Can Swarm Intelligence Match Evolution?

**Course:** Bio-Inspired Learning — UPM, MUTSC  
**Report format:** IEEE

---

## Research Question

> Can PSO, a swarm intelligence algorithm, match or surpass the computational capacity
> of real biological connectomes on memory and chaotic prediction tasks —
> and if it does, does it rediscover the same graph structure that
> 500 million years of evolution built?

---

## Project Structure

```
project/
├── PROJECT_BRIEF.md
├── data/
│   ├── fly/          (conn.npy, labels.npy)
│   ├── mouse/
│   ├── rat/
│   └── macaque/
├── images/
│   ├── 01_matrix_heatmaps.png
│   ├── 02_graph_viz.png
│   ├── 03_degree_distributions.png
│   ├── 04_memory_baseline.png
│   ├── 05_lorenz_baseline.png
│   ├── 06_pso_convergence.png
│   ├── 07_pso_vs_bio.png
│   ├── 08_graph_metrics_comparison.png
│   └── 09_complexity_ladder.png
├── notebooks/
│   ├── 01_download_and_explore.ipynb
│   ├── 02_baseline_tasks.ipynb
│   ├── 03_pso_optimisation.ipynb
│   └── 04_graph_analysis.ipynb
└── report/
    └── ieee_report.tex
```

---

## Species

| Species | Nodes | Reference |
|---------|-------|-----------|
| Fruit fly (Drosophila) | 49 | Chiang et al. 2011 |
| Mouse | 213 | Rubinov et al. 2015 |
| Rat | 503 | Bota et al. 2015 |
| Macaque | 383 | Modha & Singh 2010 |

---

## Workflow

1. **Notebook 01** — Load / generate connectomes, explore statistics, visualise
2. **Notebook 02** — Baseline reservoir performance (Memory Capacity + Lorenz)
3. **Notebook 03** — PSO optimisation (from biological wiring vs from scratch)
4. **Notebook 04** — Graph analysis before/after PSO

---

## Dependencies

```bash
pip install numpy scipy matplotlib networkx pyswarms pandas seaborn
pip install reservoirpy scikit-learn bctpy jupyter
# conn2res:
git clone https://github.com/netneurolab/conn2res.git
cd conn2res && pip install --no-deps .
```

---

## Color Convention (all figures)

| Condition | Color |
|-----------|-------|
| Biological | `#2196F3` (blue) |
| PSO from biology | `#4CAF50` (green) |
| PSO from random | `#FF5722` (orange) |
| Random null | `#9E9E9E` (grey) |
