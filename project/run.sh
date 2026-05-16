#!/bin/bash
# ============================================================
# Wild Reservoirs — Full Pipeline
# Bio-Inspired Optimisation of Neural Connectomes (Worm → Human)
# ============================================================
# Run from the project/ directory:
#   cd /path/to/project && bash run.sh
#
# Steps:
#   00. fetch_real_connectomes.py  — download 6 biological connectomes
#                                    (C. elegans, fly, mouse, rat,
#                                     macaque-weighted, human)
#   01. step01_explore.py          — visualise connectome matrices + degree distributions
#   02. step02_baselines.py        — MC + Lorenz baselines (10 runs, RidgeCV)
#   03. step03_pso.py              — PSO/DE/GWO/WOA × MC/Lorenz/NARMA-10/Mackey-Glass
#                                    (10 runs × 4 algorithms × 4 tasks × 6 species)
#   06. step06_publication_figures.py — publication-quality figures (heatmap, radar,
#                                    network graphs, violin, correlation, dot plot)
#
# NOTE: step04_analysis.py and step05_summary.py need updating to the
#       new CSV format (opt_results.csv) — skipped for now.
#
# Estimated runtime on NVIDIA RTX 6000 Ada (CUDA):
#   step02: ~5 min
#   step03: ~3-5 hours  (4 algs × 4 tasks × 6 species × 10 runs × 1000 evals)
#   step06: ~2 min
#   TOTAL:  ~3-5 hours
#
# Without GPU (CPU only): ~5-10x longer for step03.
# ============================================================

set -e

PYTHON="../venv/bin/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

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

PIPELINE_START=$(date +%s)
echo "============================================================"
echo "  Wild Reservoirs — Full Pipeline"
echo "  $(date)"
echo "============================================================"
echo "  6 species: C. elegans, fly, mouse, rat, macaque (weighted), human"
echo "  4 algorithms: PSO, DE, GWO, WOA"
echo "  4 tasks: MC, Lorenz, NARMA-10, Mackey-Glass"
echo "  10 runs per condition  |  held-out evaluation  |  GPU batch"
echo "============================================================"

run_step "00" "fetch_real_connectomes.py" \
    "Download biological connectomes (C. elegans, fly, mouse, rat, macaque-W, human)"

run_step "01" "step01_explore.py" \
    "Visualise connectomes → images/01_matrix_heatmaps.png, 02_degree_distributions.png"

run_step "02" "step02_baselines.py" \
    "Biological baselines: MC + Lorenz (10 runs, rho=0.97, RidgeCV) → data/baseline_results.csv"

run_step "03" "step03_pso.py" \
    "PSO/DE/GWO/WOA × MC/Lorenz/NARMA-10/MG × 6 species × 10 runs → data/opt_results.csv"

run_step "06" "step06_publication_figures.py" \
    "Publication figures: heatmap, radar, network graphs, violin, correlation, dot plot"

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
echo "  Data:"
echo "    data/baseline_results.csv  — biological baselines (step02)"
echo "    data/opt_results.csv       — all algorithm × task × species × run scores"
echo ""
echo "  Figures:"
ls images/*.png 2>/dev/null | sed 's/^/    /' || echo "    (none yet)"
echo ""
echo "  Logs:  logs/step00.log … logs/step06.log"
echo "  NOTE:  step04 (stats) and step05 (summary) need CSV format update — run manually."
echo "============================================================"
