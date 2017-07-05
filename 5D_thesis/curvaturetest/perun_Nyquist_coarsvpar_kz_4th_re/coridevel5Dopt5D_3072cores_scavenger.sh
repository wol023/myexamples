#!/bin/bash -l
#SBATCH --partition regular
#SBATCH --qos=scavenger
#SBATCH --nodes 96
#SBATCH --time=23:59:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 3072 ../../CORIcogent5Dopt5D.ex kinetic.in
