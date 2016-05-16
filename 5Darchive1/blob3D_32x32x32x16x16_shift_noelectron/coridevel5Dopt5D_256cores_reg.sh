#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --nodes 8
#SBATCH --time=05:59:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 256 ../../CORIcogent5Dopt5D.ex kinetic.in
