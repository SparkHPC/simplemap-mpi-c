CC=mpicc
EXEC=simplemap

CFLAGS  += -DTHETA
LDFLAGS +=

all: $(EXEC)

$(EXEC): mpi_benchmark.o
	$(CC) -g -o  $@ $^ $(LDFLAGS)

main.o: mpi_benchmark.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

clean:
	rm *.o *~ $(EXEC)
