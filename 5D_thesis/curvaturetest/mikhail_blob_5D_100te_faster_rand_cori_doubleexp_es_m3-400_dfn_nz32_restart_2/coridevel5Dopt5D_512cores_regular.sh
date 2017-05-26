#!/bin/bash -l
#SBATCH --partition=regular
#SBATCH --nodes=16
#SBATCH --time=15:30:00
#SBATCH --license=SCRATCH
#SBATCH --constraint=haswell

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 512 ../../../CORIcogent5Dopt5D.ex kinetic.in
