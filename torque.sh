#!/bin/bash
#PBS -l nodes=1:ppn=4,mem=1gb,walltime=00:10:00
#PBS -M jd.prada11@uniandes.edu.co
#PBS -m abe
#PBS -N unsupervised-learning-lab-pi

# Parallel π integration (MPI rectangle rule).
# Local:  chmod u+x torque.sh && ./torque.sh
# Cluster: qsub torque.sh

module load mpich/3.2.1 2>/dev/null || true

ROOT="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT/mpi"

make clean
make Listing12.7
mpirun -n 4 ./Listing12.7 2000000
