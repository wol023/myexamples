#!/bin/bash
#SBATCH -N 1
#SBATCH -p regular
#SBATCH -C knl,quad,cache
#SBATCH -t 02:00:00

#this is the KNL script
echo ncores,nht,arch,time > threadscale_knl.csv
for nc in 1 2 4 8 16 32 64; do
for nht in 1 2 4; do

export OMP_NUM_THREADS=$(( ${nc} * ${nht} ))
export OMP_PLACES=cores“(${nc})“
export OMP_PROC_BIND=spread

srun -n 1 -c 272 --cpu_bind=cores numactl -p 1 ../../../../KNLcogent5Dopt5D.ex kinetic.in > output
timing=<extract timing from output>

echo ${nt},${nht},knl,${timing} >> threadscale_knl.csv
done
done

