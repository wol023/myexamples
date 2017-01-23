#!/bin/bash -l
#SBATCH --partition=regular
#SBATCH --qos=premium
#SBATCH --nodes=32
#SBATCH --time=29:30:00
#SBATCH --license=SCRATCH
#SBATCH --constraint=haswell

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 1024 ../../../CORIcogent5Dopt5D.ex kinetic.in
