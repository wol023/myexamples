#PBS -N cogent5d
#PBS -V
#PBS -l nodes=6:ppn=32
#PBS -l walltime=288:00:00

cd $PBS_O_WORKDIR

mpirun ../../PERUNcogent5Dopt5D.ex kinetic.in >& perunoutput.out

