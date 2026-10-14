#!/bin/bash
#PBS -l nodes=1:ppn=4,mem=1gb,walltime=00:20:00
#PBS -M jd.prada11@uniandes.edu.co
#PBS -m abe
#PBS -N unsupervised-learning-lab-mc

# Monte Carlo convergence study + error plot.
# Local:  chmod u+x torque2.sh && ./torque2.sh
# Cluster: qsub torque2.sh

module load mpich/3.2.1 2>/dev/null || true

ROOT="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"

bash mpi/scripts/run_monte_carlo.sh
