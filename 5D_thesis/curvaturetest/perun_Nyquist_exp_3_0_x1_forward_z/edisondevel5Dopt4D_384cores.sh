#!/bin/bash
#SBATCH -N 16
#SBATCH -p debug
#SBATCH -t 00:30:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior

srun -n 384 -c 2 ../../../EDISONcogent5Dopt4D.ex kinetic.in


