#!/bin/bash -l
#SBATCH --partition=regular
#SBATCH --qos=premium
#SBATCH --nodes=8
#SBATCH --time=01:00:00
#SBATCH --license=SCRATCH
#SBATCH --constraint=haswell

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 8 ../../../CORIcogent5Dopt5D.ex kinetic.in
