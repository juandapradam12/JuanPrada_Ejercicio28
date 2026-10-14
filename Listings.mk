# Backward-compatible entry point.
# Usage: make -f Listings.mk
# Prefer: make -C mpi

.PHONY: all clean Listing12.7 integra run-pi run-mc convergence scaling vr

all clean Listing12.7 integra run-pi run-mc convergence scaling vr:
	$(MAKE) -C mpi $@
