#!/bin/sh
# Test three short example cases for branch and trunk and h5diff the chk files.
# One need to prepare 6 directories (gamsmall, gamsmall_ref, neosmall, neosmall_ref, snwithE, snwithE_ref) with input files in each directories.
# The directories to the cogent .ex files should be edited accordingly.

# gam test
cd gam_branch_renorm
source ./cleandata
mpirun -np 2 /home/wonjae/ws/branchinESL/devel5Dopt4D_merging_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex gam.in |tee output.txt
cd ../

cd gam_trunk_renorm
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/trunk_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex gam.in |tee output.txt
cd ../


# h5diff
echo "####" |tee gam_result.txt
echo "1, /home/wonjae/ws/branchinESL/devel5Dopt4D_merging_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a gam_result.txt
echo "2, /home/wonjae/ws/trunkESL/trunk_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a gam_result.txt
echo "####h5diff gam_branch_renorm vs gam_trunk_renorm chk file" |tee -a gam_result.txt
h5diff gam_branch_renorm/chk0010.4d.hdf5 gam_trunk_renorm/chk0010.4d.hdf5 |tee -a gam_result.txt
echo "####h5diff gam_branch_renorm vs gam_trunk_renorm plt file" |tee -a gam_result.txt
h5diff gam_branch_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 gam_trunk_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 |tee -a gam_result.txt
echo "####h5diff gam_branch_renorm vs gam_trunk_renorm plt map file" |tee -a gam_result.txt
h5diff gam_branch_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 gam_trunk_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 |tee -a gam_result.txt
echo "####" |tee -a gam_result.txt

