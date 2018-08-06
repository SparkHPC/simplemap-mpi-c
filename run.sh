#!/bin/bash

mpirun -f $COBALT_NODEFILE -n $1 $2 -n $3 -c $4 -b $5 -k $6 -j $7
