/*
 * Parallel Monte Carlo integration with MPI.
 *
 * Estimates I = ∫_0^1 exp(x) * sin(x) dx
 * Closed form: I = 0.5 * (e*(sin(1) - cos(1)) + 1)
 *
 * Build:  mpicc integra.c -o integra -lm
 * Run:    mpiexec -n <procs> ./integra <N_samples>
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

static double f(double x)
{
	return exp(x) * sin(x);
}

/* Exact value of ∫_0^1 e^x sin(x) dx */
static double analytical(void)
{
	return 0.5 * (M_E * (sin(1.0) - cos(1.0)) + 1.0);
}

static double monte_carlo(long long n_local, unsigned int seed)
{
	long long i;
	double sum = 0.0;

	for (i = 0; i < n_local; i++) {
		double u = rand_r(&seed) / (RAND_MAX + 1.0);
		sum += f(u);
	}
	return sum;
}

int main(int argc, char *argv[])
{
	int rank, size;
	long long n_total, n_local, remainder;
	double local_sum, global_sum, estimate, exact, rel_err;
	double t0, t1;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc < 2) {
		if (rank == 0) {
			fprintf(stderr, "Usage: %s <N_samples>\n", argv[0]);
		}
		MPI_Finalize();
		return 1;
	}

	n_total = atoll(argv[1]);
	if (n_total <= 0) {
		if (rank == 0) {
			fprintf(stderr, "N_samples must be positive.\n");
		}
		MPI_Finalize();
		return 1;
	}

	n_local = n_total / size;
	remainder = n_total % size;
	if (rank < remainder) {
		n_local += 1;
	}

	t0 = MPI_Wtime();
	local_sum = monte_carlo(n_local, (unsigned int)(time(NULL) + rank * 9973));
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	t1 = MPI_Wtime();

	if (rank == 0) {
		estimate = global_sum / (double)n_total;
		exact = analytical();
		rel_err = fabs(estimate - exact) / fabs(exact) * 100.0;
		printf("%.16f\n", estimate);
		fprintf(stderr,
			"N=%lld procs=%d estimate=%.10f exact=%.10f rel_error=%.6f%% time=%.4fs\n",
			n_total, size, estimate, exact, rel_err, t1 - t0);
	}

	MPI_Finalize();
	return 0;
}
