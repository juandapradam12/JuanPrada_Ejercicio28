# MPI programs

## Build

```bash
make -C mpi all
```

Use **Open MPI** when possible (`mpicc` / `mpirun`). On some containers MPICH’s `mpiexec` can start processes that each see `MPI_COMM_WORLD` size 1; if that happens, switch to Open MPI or set `MPIEXEC=mpirun`.

## π estimator — `Listing12.7`

```bash
mpirun -n 4 ./mpi/Listing12.7 1000000
# or
make -C mpi run-pi
```

Interactive mode (original textbook style): run without arguments and enter interval counts; `0` quits.

## Monte Carlo integrator — `integra`

```bash
mpirun -n 4 ./mpi/integra 1000000
make -C mpi convergence   # N-sweep + figures/monte_carlo_relative_error.png
```

Stdout of `integra` is the estimate only (easy to capture in files). Diagnostics go to stderr.

## Cluster scripts

| Script | Purpose |
| --- | --- |
| `torque.sh` | PBS job for π |
| `torque2.sh` | PBS / local wrapper for the MC convergence study |
| `mpi/scripts/run_monte_carlo.sh` | Portable local runner |
