#!/usr/bin/env bash
# Reproduce every number and figure in the README (~15 minutes on CPU).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> tests"
pytest -q

echo "==> headline comparisons"
python -m experiments.run_headline --dataset mackey_glass    --seeds 5
python -m experiments.run_headline --dataset nonlinear_mixed --seeds 5

echo "==> 90-configuration sweep (sharded by membership family)"
for M in gaussian gbell dsig triangular trapezoidal; do
  python -m experiments.run_sweep --dataset mackey_glass --folds 5 --mfs "$M" --out "sweep_part_$M"
done
python scripts/merge_sweep.py --out sweep_mackey_glass.csv

echo "==> figures and rule dump"
python -m experiments.make_figures --dataset mackey_glass --dims 0 3

echo "done -- see results/"
