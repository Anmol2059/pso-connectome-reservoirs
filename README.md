<div align="center">

# 🧠 Wild Reservoirs

### *Bio-Inspired Swarm & Evolutionary Optimisation of Neural Connectomes — from Worm to Human*

**Can algorithms born from animal behaviour rediscover what evolution wired into animal brains?**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Institution](https://img.shields.io/badge/UPM-ETSI%20Telecom-orange)](https://www.upm.es)

</div>

---

## 🌍 The Big Picture

Real brains are wired by **500 million years of evolution**.
Swarm optimisers are wired by **animal behaviour** — flocking birds, mutating genes, hunting wolves.

We plug **7 real biological connectomes** directly into reservoir computers, then let three
bio-inspired optimisers loose on the weights. Who wins — nature or the algorithm?

> 📄 **Full experimental details →** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

---

## 🐾 Seven Real Brains

| | Animal | Nodes | Paper |
|--|--------|------:|-------|
| 🪱 | *C. elegans* — roundworm | 279 | [Varshney et al. 2011](https://doi.org/10.1371/journal.pcbi.1001066) |
| 🪰 | *Drosophila* — fruit fly | 49 | [Chiang et al. 2011](https://doi.org/10.1016/j.cub.2010.11.056) |
| 🐭 | Mouse cortex | 112 | [Rubinov et al. 2015](https://doi.org/10.1073/pnas.1420315112) |
| 🐀 | Rat cortex | 73 | [Bota et al. 2015](https://doi.org/10.1073/pnas.1504394112) |
| 🐒 | Macaque — binary | 242 | [Modha & Singh 2010](https://doi.org/10.1073/pnas.1008054107) |
| 🐒 | Macaque — weighted (FLNe) | 29 | [Markov et al. 2014](https://doi.org/10.1093/cercor/bhs270) |
| 🧑 | Human cortex (DTI) | 83 | [Cammoun et al. 2012](https://doi.org/10.1016/j.jneumeth.2011.09.031) |

All connectomes are real measurements — downloaded automatically via [`netneurotools`](https://github.com/netneurolab/netneurotools). No synthetic data.

---

## 🦾 Three Bio-Inspired Optimisers

| | Algorithm | Inspired by | Paper |
|--|-----------|-------------|-------|
| 🐦 | **PSO** — Particle Swarm Optimisation | Bird flocking & fish schooling | [Kennedy & Eberhart 1995](https://doi.org/10.1109/ICNN.1995.488968) |
| 🧬 | **DE** — Differential Evolution | Genetic mutation & recombination | [Storn & Price 1997](https://doi.org/10.1023/A:1008202821328) |
| 🐺 | **GWO** — Grey Wolf Optimiser | Wolf pack hunting hierarchy | [Mirjalili et al. 2014](https://doi.org/10.1016/j.advengsoft.2013.12.007) |

Each algorithm gets **1 000 objective evaluations** (20 particles × 50 iterations) per run,
evaluated in parallel on GPU via batched `torch.bmm`.

---

## 🧪 What We Measure

| Task | Question | Metric |
|------|----------|--------|
| 💾 **Memory Capacity** | How many past inputs can the reservoir recall? | MC = Σ R² across 50 lags — higher is better |
| 🌀 **Lorenz Prediction** | Can the reservoir forecast chaotic dynamics? | NRMSE — lower is better |

---

## ⚡ Quick Start

```bash
git clone https://github.com/Anmol2059/pso-connectome-reservoirs.git
cd pso-connectome-reservoirs

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install conn2res (ESN library)
git clone https://github.com/netneurolab/conn2res.git
cd conn2res && pip install --no-deps . && cd ..

# Run everything (~70-100 min on GPU)
cd project && bash run.sh
```

---

## 🗂️ Pipeline at a Glance

```
bash run.sh
├── 00  fetch_real_connectomes.py  →  download 7 connectomes
├── 01  step01_explore.py          →  visualise networks
├── 02  step02_baselines.py        →  biological MC + Lorenz baselines
├── 03  step03_pso.py              →  PSO 🐦 + DE 🧬 + GWO 🐺 optimisation
├── 04  step04_analysis.py         →  stats, Cohen's d, graph metrics
└── 05  step05_summary.py          →  full report → data/summary.txt
```

---

## 💻 Runtime

| Hardware | Estimated Time |
|----------|:--------------:|
| NVIDIA GPU (CUDA) | ~70–100 min |
| Apple Silicon (MPS) | ~90–120 min |
| CPU only | ~5–7 hr |

---

## 🙏 Acknowledgements

We thank the neuroscience teams who made their data openly available:

[Varshney 2011](https://doi.org/10.1371/journal.pcbi.1001066) · [Chiang 2011](https://doi.org/10.1016/j.cub.2010.11.056) · [Rubinov 2015](https://doi.org/10.1073/pnas.1420315112) · [Bota 2015](https://doi.org/10.1073/pnas.1504394112) · [Modha 2010](https://doi.org/10.1073/pnas.1008054107) · [Markov 2014](https://doi.org/10.1093/cercor/bhs270) · [Cammoun 2012](https://doi.org/10.1016/j.jneumeth.2011.09.031) · [netneurotools](https://github.com/netneurolab/netneurotools) · [conn2res](https://github.com/netneurolab/conn2res)

---

<div align="center">

**Anmol Guragain · Savvas Kakalis · Juan Ignacio Godino Llorente**

*ETSI de Telecomunicación, Universidad Politécnica de Madrid*

</div>
