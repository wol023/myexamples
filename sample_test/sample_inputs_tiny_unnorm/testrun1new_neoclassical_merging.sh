#!/bin/sh
# Test three short example cases for branch and trunk and h5diff the chk files.
# One need to prepare 6 directories (gamsmall, gamsmall_ref, neosmall, neosmall_ref, snwithE, snwithE_ref) with input files in each directories.
# The directories to the cogent .ex files should be edited accordingly.

cd neoclassical_branch_unnorm
source ./cleandata
mpirun -np 2 /home/wonjae/ws/branchinESL/devel5Dopt4D_merging_unnorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex neoclass_lin.in |tee output.txt
cd ../

cd neoclassical_trunk_unnorm
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/trunk_unnorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex neoclass_lin.in |tee output.txt
cd ../


# h5diff
echo "####" |tee neoclassical_result.txt
echo "1, /home/wonjae/ws/branchinESL/devel5Dopt4D_merging_unnorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a neoclassical_result.txt
echo "2, /home/wonjae/ws/trunkESL/trunk_unnorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a neoclassical_result.txt
echo "####h5diff neoclassical_branch_unnorm vs neoclassical_trunk_unnorm chk file" |tee -a neoclassical_result.txt
h5diff neoclassical_branch_unnorm/chk0010.4d.hdf5 neoclassical_trunk_unnorm/chk0010.4d.hdf5 |tee -a neoclassical_result.txt
echo "####h5diff neoclassical_branch_unnorm vs neoclassical_trunk_unnorm plt file" |tee -a neoclassical_result.txt
h5diff neoclassical_branch_unnorm/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 neoclassical_trunk_unnorm/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 |tee -a neoclassical_result.txt
echo "####h5diff neoclassical_branch_unnorm vs neoclassical_trunk_unnorm plt map file" |tee -a neoclassical_result.txt
h5diff neoclassical_branch_unnorm/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 neoclassical_trunk_unnorm/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 |tee -a neoclassical_result.txt
echo "####" |tee -a neoclassical_result.txt

