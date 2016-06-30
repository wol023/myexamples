#PBS -N cogent
#PBS -V
#PBS -l nodes=4:ppn=16
cd $PBS_O_WORKDIR

#/usr/local/openmpi/bin/mpirun /home/nfs/wol023/esl/branchESL/devel5D/exec/cogent.Linux.64.mpiCC.gfortran.OPT.MPI.ex  GamSmall.in >& output.out
/usr/local/openmpi/bin/mpirun /home/nfs/wol023/esl/trunkESL/trunk/exec/cogent.Linux.64.mpiCC.gfortran.OPT.MPI.ex  kinetic.in >& output.out

