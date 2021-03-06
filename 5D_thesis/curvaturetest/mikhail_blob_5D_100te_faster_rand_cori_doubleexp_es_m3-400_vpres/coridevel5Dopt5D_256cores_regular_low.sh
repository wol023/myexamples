#!/bin/bash -l
#SBATCH --partition=regular
#SBATCH --qos=low
#SBATCH --nodes=8
#SBATCH --time=22:30:00
#SBATCH --license=SCRATCH
#SBATCH --constraint=haswell

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 256 ../../../CORIcogent5Dopt5D.ex kinetic.in
