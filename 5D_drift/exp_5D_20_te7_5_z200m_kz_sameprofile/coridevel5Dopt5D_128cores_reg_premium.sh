#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --qos=premium
#SBATCH --nodes 4
#SBATCH --time=05:59:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 128 ../../CORIcogent5Dopt5D.ex kinetic.in
