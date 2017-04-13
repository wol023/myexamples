#!/bin/sh
# Test three short example cases for branch and trunk and h5diff the chk files.
# One need to prepare 6 directories (gamsmall, gamsmall_ref, neosmall, neosmall_ref, snwithE, snwithE_ref) with input files in each directories.
# The directories to the cogent .ex files should be edited accordingly.

cd sn_branch_renorm
source ./cleandata
mpirun -np 2 /home/wonjae/ws/branchinESL/devel5Dopt4D_merging_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex sn_loss_cone_ten_block.in |tee output.txt
cd ../

cd sn_trunk_renorm
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/trunk_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex sn_loss_cone_ten_block.in |tee output.txt
cd ../


# h5diff
echo "####" |tee sn_result.txt
echo "1, /home/wonjae/ws/branchinESL/devel5Dopt4D_merging_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a sn_result.txt
echo "2, /home/wonjae/ws/trunkESL/trunk_renorm/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a sn_result.txt
echo "####h5diff sn_branch_renorm vs sn_trunk_renorm chk file" |tee -a sn_result.txt
h5diff sn_branch_renorm/chk0010.4d.hdf5 sn_trunk_renorm/chk0010.4d.hdf5 |tee -a sn_result.txt
echo "####h5diff sn_branch_renorm vs sn_trunk_renorm plt file" |tee -a sn_result.txt
h5diff sn_branch_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 sn_trunk_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 |tee -a sn_result.txt
echo "####h5diff sn_branch_renorm vs sn_trunk_renorm plt map file" |tee -a sn_result.txt
h5diff sn_branch_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 sn_trunk_renorm/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 |tee -a sn_result.txt
echo "####" |tee -a sn_result.txt

