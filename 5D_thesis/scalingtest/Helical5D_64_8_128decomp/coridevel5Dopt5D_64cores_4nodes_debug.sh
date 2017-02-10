#!/bin/bash -l
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --time=00:30:00
#SBATCH --license=SCRATCH
#SBATCH --constraint=haswell

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 64 ../../../CORIcogent5Dopt5D.ex kinetic.in
