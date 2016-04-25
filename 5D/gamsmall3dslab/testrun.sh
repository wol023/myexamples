#!/bin/sh

echo "rm -v chk*.?d.hdf5"
rm -v chk*.?d.hdf5 
echo "rm -v pout.*"
rm -v pout.*

# gam test
mpirun -np 1 /home/wonjae/ws/branchinESL/devel5Ddbg5D_beforemerge/exec/cogent.Linux.64.mpicxx.gfortran.DEBUG.MPI.ex  GamSmall.in |tee output.txt
#mpirun -np 1 /home/wonjae/ws/branchinESL/devel5Ddbg5D/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex  GamSmall.in |tee output.txt

