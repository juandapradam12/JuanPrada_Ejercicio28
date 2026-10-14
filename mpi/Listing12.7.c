/*
 * Parallel numerical integration for π (MPI rectangle rule).
 * Estimates π ≈ ∫_0^1 4 / (1 + x^2) dx
 *
 * Build:  mpicc Listing12.7.c -o Listing12.7 -lm
 * Run:    mpiexec -n <procs> ./Listing12.7 [n_intervals]
 *         If n_intervals is omitted, prompts interactively (0 quits).
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	const double PI_VALUE = 3.141592653589793238462643;
	int rank, size, i, len, n;
	int interactive;
	double mypi, pi, h, sum, x;
	double start_time = 0.0, end_time = 0.0, computation_time;
	char name[80];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(name, &len);

	interactive = (argc < 2);

	do {
		if (rank == 0) {
			if (interactive) {
				printf("Enter the number of intervals: (0 quits) ");
				fflush(stdout);
				if (scanf("%d", &n) != 1) {
					n = 0;
				}
			} else {
				n = atoi(argv[1]);
			}
			start_time = MPI_Wtime();
		}

		MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (n == 0) {
			break;
		}

		h = 1.0 / (double)n;
		sum = 0.0;
		for (i = rank + 1; i <= n; i += size) {
			x = h * ((double)i - 0.5);
			sum += 4.0 / (1.0 + x * x);
		}
		mypi = h * sum;

		printf("This is my sum: %.16f from rank: %d in: %s\n", mypi, rank, name);
		MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			end_time = MPI_Wtime();
			computation_time = end_time - start_time;
			printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI_VALUE));
			printf("Time of calculating pi is: %f\n", computation_time);
		}
	} while (interactive);

	MPI_Finalize();
	return 0;
}
