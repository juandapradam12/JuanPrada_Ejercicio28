CC      = mpicc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm
# Prefer mpirun (Open MPI / MPICH). Override with: make MPIEXEC='mpiexec -n'
MPIEXEC ?= mpirun
NP      ?= 4

.PHONY: all clean run-pi run-mc convergence scaling

all: Listing12.7 integra

Listing12.7: Listing12.7.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

integra: integra.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run-pi: Listing12.7
	$(MPIEXEC) -n $(NP) ./Listing12.7 1000000

run-mc: integra
	$(MPIEXEC) -n $(NP) ./integra 1000000

# Convergence study used by plots.py
convergence: integra
	@mkdir -p ../outputs
	@for n in 100 1000 10000 100000 1000000 10000000; do \
		echo "N=$$n"; \
		$(MPIEXEC) -n $(NP) ./integra $$n > ../outputs/mc_$$n.txt; \
	done
	python3 plots.py

# Strong scaling: wall time / speedup vs ranks
# Quote MPIEXEC so values like `mpirun --allow-run-as-root` survive Make/shell.
scaling: integra
	MPIEXEC="$(MPIEXEC)" python3 scaling.py --mpiexec "$(MPIEXEC)" --procs 1 2 4 --n-samples 5000000 --repeats 2

clean:
	rm -f Listing12.7 integra ../outputs/mc_*.txt ../outputs/mpi_scaling.csv
