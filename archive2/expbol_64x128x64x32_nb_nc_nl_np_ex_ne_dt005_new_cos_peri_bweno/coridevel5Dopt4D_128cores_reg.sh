#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --nodes 4
#SBATCH --time=05:30:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 128 ../CORIcogent5Dopt4D.ex kinetic.in
