#!/bin/bash
#SBATCH -N 1
#SBATCH -p regular
#SBATCH -C knl,quad,cache
#SBATCH -t 02:00:00

#this is the KNL script
echo ncores,nht,arch,time > threadscale_knl.csv
#for nc in 64 32 16 8 4 2 1; do
for nc in 64 32 16; do
for nht in 1 2 4; do

export OMP_NUM_THREADS=$(( ${nc} * ${nht} ))
export OMP_PLACES=cores"(${nc})"
#export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 1 -c 272 --cpu_bind=cores ../../../../KNLcogent5Dopt5D.ex kinetic.in > output 
t2=$(cat output | grep "Step 2 completed" | awk '{print $12}')
t1=$(cat output | grep "Step 1 completed" | awk '{print $12}')
timing=`echo "$t2 - $t1" | bc`

echo ${nt},${nht},knl,${timing} >> threadscale_knl.csv
done
done

