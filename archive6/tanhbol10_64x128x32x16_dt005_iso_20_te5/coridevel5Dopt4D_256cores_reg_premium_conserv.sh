#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --qos=premium
#SBATCH --nodes 8
#SBATCH --time=03:59:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 256 ../CORIcogent5Dopt4D_conserv.ex kinetic.in
