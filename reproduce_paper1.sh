#!/usr/bin/env bash
set -e

echo "================================================================="
echo "Paper 1 Reproducibility Pipeline (Chaos, Solitons & Fractals)"
echo "Search-Budget Artifacts in Continuous Causal Emergence"
echo "================================================================="

mkdir -p experiments figures evidence .cache/matplotlib .cache/fontconfig

# Matplotlib and fontconfig need writable cache directories.  Keeping them
# inside the repository makes the pipeline work on machines where the default
# locations under the home directory are not writable.
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLBACKEND=Agg

if [ ! -f data/ff30_daily_returns.csv ]; then
  echo "data/ff30_daily_returns.csv is missing."
  echo "The industry portfolio returns are not redistributed with this"
  echo "repository.  Retrieve them from the source with:"
  echo
  echo "    python3 data/download_ff30.py"
  echo
  echo "See data/README.md for why."
  exit 1
fi

echo "[1/7] Analytical ground-truth benchmark (Table I)..."
python3 experiments/run_analytical_benchmarks.py

echo "[2/7] Operator scaling and normalization control (Fig. 1)..."
python3 experiments/run_operator_scaling.py

echo "[3/7] Macro-dimension corruption (Fig. 3)..."
python3 experiments/run_dimension_corruption.py

echo "[4/7] Non-financial generalization: coupled network, sample size, climate (Fig. 4a)..."
python3 experiments/run_nonfinancial_generalization.py

echo "[5/7] Learning rate versus iteration budget sweep (Fig. 4b)..."
python3 experiments/run_lr_vs_iter_sweep.py

echo "[6/7] Feasible-witness suboptimality certificate on the empirical trajectory (Fig. 2)..."
python3 experiments/run_diagnostic_bracket.py

echo "[7/7] Generating publication-quality figures (Figs. 1-4)..."
python3 experiments/plot_all_figures.py

echo "================================================================="
echo "REPRODUCIBILITY PIPELINE COMPLETED"
echo "Figures:    figures/"
echo "Tables:     experiments/  and  evidence/q1_bounds_summary.csv"
echo "================================================================="
