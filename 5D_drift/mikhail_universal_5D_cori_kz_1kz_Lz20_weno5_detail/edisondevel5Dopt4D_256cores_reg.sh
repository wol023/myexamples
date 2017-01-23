#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --nodes 11
#SBATCH --time=24:59:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 256 ../EDISONcogent5Dopt4D.ex kinetic.in
