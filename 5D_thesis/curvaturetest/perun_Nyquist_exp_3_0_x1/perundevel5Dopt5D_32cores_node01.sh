#PBS -N cogent5d
#PBS -V
#PBS -l nodes=node01:ppn=32
#PBS -l walltime=16:00:00

cd $PBS_O_WORKDIR

mpirun ../../../PERUNcogent5Dopt5D.ex kinetic.in >& perunoutput.out

