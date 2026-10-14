# Backward-compatible entry point.
# Usage: make -f Listings.mk
# Prefer: make -C mpi

.PHONY: all clean Listing12.7 integra run-pi run-mc convergence

all clean Listing12.7 integra run-pi run-mc convergence:
	$(MAKE) -C mpi $@
