#PBS -N cogent5d
#PBS -V
#PBS -l nodes=node01:ppn=32+node02:ppn=32+node04:ppn=32+node05:ppn=32+node06:ppn=32+node07:ppn=32+node10:ppn=32+node12:ppn=32

#PBS -l walltime=076:00:00

cd $PBS_O_WORKDIR

mpirun ../../PERUNcogent5Dopt5D.ex kinetic.in >& perunoutput.out

