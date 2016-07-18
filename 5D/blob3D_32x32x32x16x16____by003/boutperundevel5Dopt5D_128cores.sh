#PBS -N cogent
#PBS -V
#PBS -l nodes=4:ppn=128
cd #PBS_O_WORKDIR

/usr/local/openmpi/bin/mpirun ../PERUNcogent5Dopt5D.ex kinetic.in
