#!/bin/bash -l
#SBATCH --partition debug
#SBATCH --nodes 24
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 768 ../../CORIcogent5Dopt5D.ex kinetic.in
