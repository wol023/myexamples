#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --qos=premium
#SBATCH --nodes 22
#SBATCH --time=03:30:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 512 ../EDISONcogent5Dopt4D.ex kinetic.in
