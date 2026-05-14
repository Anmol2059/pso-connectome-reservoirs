#!/bin/bash
# ============================================================
# Wild Reservoirs — Full Pipeline
# From Worm to Human: Bio-Inspired Optimisation of Neural Connectomes
# ============================================================
# Run from the project/ directory:
#   cd /path/to/project
#   bash run.sh
#
# Steps:
#   0. fetch_real_connectomes.py  — download 7 biological connectomes
#                                   (C. elegans, fly, mouse, rat,
#                                    macaque-binary, macaque-weighted, human)
#   1. step01_explore.py          — visualise connectomes
#   2. step02_baselines.py        — MC + Lorenz baselines (5 runs, RidgeCV)
#   3. step03_pso.py              — PSO/DE/GWO optimisation (5 runs × 7 species)
#                                   both MC and Lorenz tasks, held-out evaluation
#   4. step04_analysis.py         — stats + Cohen's d + graph topology metrics
#   5. step05_summary.py          — generate data/summary.txt
#
# Estimated runtime on NVIDIA RTX 6000 Ada (CUDA):
#   step02: ~5 min   (RidgeCV baselines, 7 species)
#   step03: ~60-90 min (PSO+DE+GWO × MC+Lorenz × 7 species)
#   other steps: ~3 min total
#   TOTAL: ~45-70 min
#
# Without GPU (CPU only): ~3-5x longer for step03
# ============================================================

set -e   # exit immediately on any error

PYTHON="../venv/bin/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── helper ────────────────────────────────────────────────────────────────────
run_step() {
    local step_num="$1"
    local script="$2"
    local description="$3"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step ${step_num}: ${description}"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    $PYTHON "$script" 2>&1 | tee "${LOG_DIR}/step${step_num}.log"
    echo "  ✓ Step ${step_num} done  $(date)"
}

# ── start ─────────────────────────────────────────────────────────────────────
PIPELINE_START=$(date +%s)
echo "============================================================"
echo "  PSO-Optimised Connectome Reservoirs — Full Pipeline"
echo "  $(date)"
echo "============================================================"
echo "  Scientific fixes: held-out eval, rho=0.97, RidgeCV,"
echo "  directed weighted graph metrics, Cohen's d, all-runs convergence plot"
echo "============================================================"

run_step "00" "fetch_real_connectomes.py" \
    "Download real biological connectomes (Chiang 2011, Rubinov 2015, Bota 2015, Modha 2010)"

run_step "01" "step01_explore.py" \
    "Explore & visualise connectomes → images/01_*.png, 02_*.png"

run_step "02" "step02_baselines.py" \
    "MC + Lorenz baselines (5 runs, rho=0.97, RidgeCV) → data/baseline_results.csv"

run_step "03" "step03_pso.py" \
    "PSO/DE/GWO (5 runs × 3 algorithms × 2 tasks × 7 species) → data/opt_results.csv"

run_step "04" "step04_analysis.py" \
    "Stats + Cohen's d + weighted directed graph metrics → images/06_*.png, 07_*.png"

run_step "05" "step05_summary.py" \
    "Generate LLM-ready report summary → data/summary.txt"

# ── summary ───────────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE  $(date)"
echo "  Total time: ${MINS}m ${SECS}s"
echo "============================================================"
echo ""
echo "  Results:"
echo "    data/baseline_results.csv  — MC + Lorenz per run (biological)"
echo "    data/pso_results.csv       — PSO held-out scores per run"
echo "    data/stats_summary.csv     — paired t-tests + Cohen's d"
echo "    data/graph_metrics.csv     — weighted directed topology"
echo "    data/input_sensitivity.csv — MC vs input node seed"
echo "    data/summary.txt           — LLM-ready full report text"
echo ""
echo "  Figures:"
ls images/*.png 2>/dev/null | sed 's/^/    /'
echo ""
echo "  Logs:  logs/step00.log … logs/step05.log"
echo "============================================================"
