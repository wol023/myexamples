#PBS -q debug
#PBS -l mppwidth=48
#PBS -l walltime=00:30:00

cd $PBS_O_WORKDIR   # optional, since this is the default behavior
aprun -n 32 ../../EDISONcogenttrunk.ex kinetic.in
