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
mpirun -n 4 ./mpi/integra 1000000 crude
mpirun -n 4 ./mpi/integra 1000000 antithetic
mpirun -n 4 ./mpi/integra 1000000 control
mpirun -n 4 ./mpi/integra 1000000 importance
make -C mpi convergence   # N-sweep + figures/monte_carlo_relative_error.png
make -C mpi scaling       # strong scaling → figures/mpi_strong_scaling.png
make -C mpi vr            # variance-reduction bake-off → figures/monte_carlo_variance_reduction.png
```

Stdout of `integra` is the estimate only (easy to capture in files). Diagnostics go to stderr.

| Method | Idea |
| --- | --- |
| `crude` | Uniform Monte Carlo |
| `antithetic` | Average \(f(U)\) and \(f(1-U)\) |
| `control` | Control variate with \(e^{x}\) (known mean \(e-1\)) |
| `importance` | Sample from \(g(x)=2x\) on \([0,1]\), weight by \(f/g\) |

### Strong scaling

`mpi/scaling.py` runs a fixed \(N\) across ranks `{1,2,4}`, averages wall time, and plots speedup vs ideal. Override with:

```bash
python3 mpi/scaling.py --procs 1 2 4 8 --n-samples 8000000 --repeats 3
```

## Cluster scripts

| Script | Purpose |
| --- | --- |
| `torque.sh` | PBS job for π |
| `torque2.sh` | PBS / local wrapper for the MC convergence study |
| `mpi/scripts/run_monte_carlo.sh` | Portable local runner |
