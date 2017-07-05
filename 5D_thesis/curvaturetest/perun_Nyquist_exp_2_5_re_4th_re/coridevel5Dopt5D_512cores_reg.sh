#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --nodes 16
#SBATCH --time=23:59:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 512 ../../CORIcogent5Dopt5D.ex kinetic.in
