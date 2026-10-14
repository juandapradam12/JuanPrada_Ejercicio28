#!/usr/bin/env bash
# Local (non-Torque) runner for the Monte Carlo convergence study.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/mpi"

make clean
make integra
mkdir -p "$ROOT/outputs"

PROCS="${PROCS:-4}"
MPIEXEC="${MPIEXEC:-mpirun}"
for n in 100 1000 10000 100000 1000000 10000000; do
  echo "Running Monte Carlo with N=${n} on ${PROCS} processes..."
  "$MPIEXEC" -n "$PROCS" ./integra "$n" > "$ROOT/outputs/mc_${n}.txt"
done

python3 plots.py
echo "Done. See figures/monte_carlo_relative_error.png"
