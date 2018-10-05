CC?=mpicc
EXEC=simplemap

CFLAGS  += -O3 -std=gnu99 -Wall -Wextra
LDFLAGS +=

all: $(EXEC)

$(EXEC): mpi_benchmark.o
	$(CC) -g -o  $@ $^ $(LDFLAGS)

main.o: mpi_benchmark.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

clean:
	-rm -f *.o *~ $(EXEC)
