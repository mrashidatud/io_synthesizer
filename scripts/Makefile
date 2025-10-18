CC      = mpicc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS =
LDLIBS  = -lm

all: mpi_synthio

mpi_synthio: mpi_synthio.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f mpi_synthio
