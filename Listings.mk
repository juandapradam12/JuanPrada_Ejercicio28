# compilo con make -f Listings.mk

Listing12.7: Listing12.7.c
	mpicc Listing12.7.c -o Listing12.7
	mpirun -np 2 ./Listing12.7

clean: 
	rm Listing12.7
