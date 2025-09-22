
CC=mpicc
CFLAGS=-O3 -Wall -Wextra
LDFLAGS=

all: mpi_synthio

mpi_synthio: mpi_synthio.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f mpi_synthio
