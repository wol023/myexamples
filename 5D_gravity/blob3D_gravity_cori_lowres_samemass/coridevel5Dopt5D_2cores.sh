#!/bin/bash -l
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 2 ../../CORIcogent5Dopt5D.ex kinetic.in
