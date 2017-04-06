#!/bin/bash
t2=$(cat slurm-0.out | grep "Step 2 completed" | awk '{print $12}')
t1=$(cat slurm-0.out | grep "Step 1 completed" | awk '{print $12}')
timing=`echo "$t2 - $t1" | bc`
echo step2 $t2
echo step1 $t1
echo timing $timing

