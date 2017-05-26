#!/bin/bash
#SBATCH -N 16
#SBATCH -p debug
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior

srun -n 256 -c 2 ../../../EDISONcogent5Dopt4D.ex kinetic.in


