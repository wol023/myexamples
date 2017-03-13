#!/bin/bash
#SBATCH -N 1
#SBATCH -p debug
#SBATCH -C haswell
#SBATCH -t 00:30:00

#this is the KNL script
echo ncores,nht,arch,time > threadscale_knl.csv
for nc in 4 8 16; do
for nht in 1 2; do

export OMP_NUM_THREADS=$(( ${nc} * ${nht} ))

#export OMP_PLACES=cores(${nc})
#export OMP_PROC_BIND=spread

srun -n ${OMP_NUM_THREADS}  --cpu_bind=sockets ../../../../KNLcogent5Dopt5D.ex kinetic.in > output
t2=$(cat output | grep "Step 2 completed" | awk '{print $12}')
t1=$(cat output | grep "Step 1 completed" | awk '{print $12}')
timing=`echo "$t2 - $t1" | bc`

echo ${OMP_NUM_THREADS},${nht},hsw,${timing} >> threadscale_knl.csv
done
done

