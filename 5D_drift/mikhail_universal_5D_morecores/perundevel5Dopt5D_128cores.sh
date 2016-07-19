#PBS -N cogent5d
#PBS -V
#PBS -l nodes=4:ppn=32

cd $PBS_O_WORKDIR

mpirun ../../PERUNcogent5Dopt5D.ex kinetic.in >& perunoutput.out

