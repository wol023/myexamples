#!/bin/bash
#SBATCH -N 32
#SBATCH -C knl,quad,flat
#SBATCH -p debug
#SBATCH -J cogent5d
#SBATCH -t 00:30:00
#SBATCH --license=SCRATCH

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior

#run the application:
srun -n 1024 -c 8 --cpu_bind=cores numactl -p 1 ../../../KNLcogent5Dopt5D.ex kinetic.in

