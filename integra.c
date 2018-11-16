#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

/* git commit -m "Second commit" */

/* COMPILO CON mpicc integra.c -o integra.x -lm Y EJECUTO CON mpiexec -n 4 ./integra.x N */

double f(double x)
{
	exp(x)*sin(x);
}


double n_rand(double lambda)
{
	/* por ahora uso una distribucion exponencial para integrar */
	double u;
	u = rand() / (RAND_MAX + 1.0);

	return -log(1- u) / lambda;
}

double MCI(int N)
{
	double *x;
	x = malloc(sizeof(double *)*N);

	int i; 

	for(i = 0; i < N; i++)
	{
            	x[i] = n_rand(0.05);
    	}

	double *F; 
	F = malloc(sizeof(double *)*N);

	for(i = 0; i < N; i++)
	{
            	F[i] = f(x[i]);
    	}

	double sum = 0.0;
	
	for(i = 0; i < N; i++)
	{
            	sum = sum + F[i];
    	}
	

	return sum/N; 
}

int main(int argc, char *argv[])
{

	int N = atoi(argv[1]);
	
	int rank, size;

	/* starts MPI */
	MPI_Init (&argc, &argv);
	/* get current process rank */
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	/* get number of processes */
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	/* print the rank and number of processes */
	MCI(N);
	printf("%lf", MCI(N));
	printf("\n");
	/* ends MPI */
	MPI_Finalize();

	return 0;
}
