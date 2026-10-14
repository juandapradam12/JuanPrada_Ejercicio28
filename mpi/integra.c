/*
 * Parallel Monte Carlo integration with MPI + variance reduction.
 *
 * Estimates I = ∫_0^1 exp(x) * sin(x) dx
 * Closed form: I = 0.5 * (e*(sin(1) - cos(1)) + 1)
 *
 * Build:  mpicc integra.c -o integra -lm
 * Run:    mpirun -n <procs> ./integra <N_samples> [method]
 *
 * Methods:
 *   0 / crude       — plain uniform Monte Carlo
 *   1 / antithetic  — antithetic variates: (f(U)+f(1-U))/2
 *   2 / control     — control variate using e^x (known mean e-1)
 *   3 / importance  — importance sampling with g(x)=2x on [0,1]
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

enum {
	METHOD_CRUDE = 0,
	METHOD_ANTITHETIC = 1,
	METHOD_CONTROL = 2,
	METHOD_IMPORTANCE = 3
};

static double f(double x)
{
	return exp(x) * sin(x);
}

static double analytical(void)
{
	return 0.5 * (M_E * (sin(1.0) - cos(1.0)) + 1.0);
}

static int parse_method(const char *s)
{
	if (s == NULL || strcmp(s, "0") == 0 || strcmp(s, "crude") == 0) {
		return METHOD_CRUDE;
	}
	if (strcmp(s, "1") == 0 || strcmp(s, "antithetic") == 0) {
		return METHOD_ANTITHETIC;
	}
	if (strcmp(s, "2") == 0 || strcmp(s, "control") == 0) {
		return METHOD_CONTROL;
	}
	if (strcmp(s, "3") == 0 || strcmp(s, "importance") == 0) {
		return METHOD_IMPORTANCE;
	}
	return -1;
}

static const char *method_name(int method)
{
	switch (method) {
	case METHOD_CRUDE:
		return "crude";
	case METHOD_ANTITHETIC:
		return "antithetic";
	case METHOD_CONTROL:
		return "control";
	case METHOD_IMPORTANCE:
		return "importance";
	default:
		return "unknown";
	}
}

/*
 * Each method returns a sum of n_local unbiased contributions.
 * The global estimate is total_sum / n_total.
 *
 * Antithetic counts each pair as one sample (still n_local contributions).
 * Control / importance also return n_local contributions.
 */
static double monte_carlo(long long n_local, unsigned int seed, int method)
{
	long long i;
	double sum = 0.0;
	const double mean_exp = M_E - 1.0; /* E[e^X], X~U(0,1) */
	/* Optimal-ish control coefficient from a short pilot on rank logic:
	 * c ≈ Cov(f,e^x)/Var(e^x). Fixed educational value near 0.85. */
	const double c_control = 0.85;

	for (i = 0; i < n_local; i++) {
		double u = rand_r(&seed) / (RAND_MAX + 1.0);

		if (method == METHOD_ANTITHETIC) {
			sum += 0.5 * (f(u) + f(1.0 - u));
		} else if (method == METHOD_CONTROL) {
			double fx = f(u);
			double gx = exp(u);
			sum += fx - c_control * (gx - mean_exp);
		} else if (method == METHOD_IMPORTANCE) {
			/* Sample Y from g(y)=2y on [0,1]: CDF G=y^2 => Y=sqrt(U) */
			double y = sqrt(u);
			double g = 2.0 * y;
			sum += f(y) / g;
		} else {
			sum += f(u);
		}
	}
	return sum;
}

int main(int argc, char *argv[])
{
	int rank, size, method;
	long long n_total, n_local, remainder;
	double local_sum, global_sum, estimate, exact, rel_err;
	double t0, t1;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc < 2) {
		if (rank == 0) {
			fprintf(stderr,
				"Usage: %s <N_samples> [crude|antithetic|control|importance]\n",
				argv[0]);
		}
		MPI_Finalize();
		return 1;
	}

	n_total = atoll(argv[1]);
	method = parse_method(argc >= 3 ? argv[2] : "crude");
	if (n_total <= 0 || method < 0) {
		if (rank == 0) {
			fprintf(stderr, "Invalid N_samples or method.\n");
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
	local_sum = monte_carlo(n_local, (unsigned int)(time(NULL) + rank * 9973 + 17 * method), method);
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	t1 = MPI_Wtime();

	if (rank == 0) {
		estimate = global_sum / (double)n_total;
		exact = analytical();
		rel_err = fabs(estimate - exact) / fabs(exact) * 100.0;
		printf("%.16f\n", estimate);
		fprintf(stderr,
			"method=%s N=%lld procs=%d estimate=%.10f exact=%.10f rel_error=%.6f%% time=%.4fs\n",
			method_name(method), n_total, size, estimate, exact, rel_err, t1 - t0);
	}

	MPI_Finalize();
	return 0;
}
