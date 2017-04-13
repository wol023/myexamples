#!/bin/sh
# Test three short example cases for branch and trunk and h5diff the chk files.
# One need to prepare 6 directories (gamsmall, gamsmall_ref, neosmall, neosmall_ref, snwithE, snwithE_ref) with input files in each directories.
# The directories to the cogent .ex files should be edited accordingly.

DIR_UNNORM=sample_inputs_tiny_unnorm
DIR_RENORM=sample_inputs_tiny_renorm

BRANCH_TOKEN='trunk_'
UN_TOKEN='_unnorm'
RE_TOKEN='_renorm'

REV_NUM='r896'
#REV_NUM='r960'
#REV_NUM='r991'
#REV_NUM='r1085'
#REV_NUM='r1088'
#REV_NUM='r1156'
#REV_NUM='r1090'
#REV_NUM='r1099'
#REV_NUM='r1102'
#REV_NUM='r1152'
#REV_NUM='r1153'
#REV_NUM='r1191'

RUN_PATH_UN=$BRANCH_TOKEN$REV_NUM$UN_TOKEN
RUN_PATH_RE=$BRANCH_TOKEN$REV_NUM$RE_TOKEN

TOTAL_RESULT_FILE="${REV_NUM}_trunk_result.txt"
echo "####" |tee $TOTAL_RESULT_FILE
echo "1, /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $TOTAL_RESULT_FILE 
echo "2, /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $TOTAL_RESULT_FILE 

#######################################################################################################################################
# gam test
TR_UN=gam_trunk_unnorm
TR_RE=gam_trunk_renorm
INPUT_FILE=gam.in
RESULT_FILE="gam_result_${REV_NUM}.txt"

echo  |tee -a $TOTAL_RESULT_FILE
echo "gam test" |tee -a $TOTAL_RESULT_FILE

cd $DIR_UNNORM/$TR_UN
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

cd $DIR_RENORM/$TR_RE
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

echo "####" |tee $RESULT_FILE
echo "1, /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE 
echo "2, /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE 

echo "####h5diff $TR_UN vs $TR_RE chk file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/chk0010.4d.hdf5 $DIR_RENORM/$TR_RE/chk0010.4d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt map file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE

#######################################################################################################################################
# neoclassical test
TR_UN=neoclassical_trunk_unnorm
TR_RE=neoclassical_trunk_renorm
INPUT_FILE=neoclass_lin.in
RESULT_FILE=neo_result_${REV_NUM}.txt

echo  |tee -a $TOTAL_RESULT_FILE
echo "neoclassical test" |tee -a $TOTAL_RESULT_FILE

cd $DIR_UNNORM/$TR_UN
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

cd $DIR_RENORM/$TR_RE
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

echo "####" |tee $RESULT_FILE
echo "1, /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE
echo "2, /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE

echo "####h5diff $TR_UN vs $TR_RE chk file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/chk0010.4d.hdf5 $DIR_RENORM/$TR_RE/chk0010.4d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt map file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE

#######################################################################################################################################
# sn test
TR_UN=sn_trunk_unnorm
TR_RE=sn_trunk_renorm
INPUT_FILE=sn_loss_cone_ten_block.in
RESULT_FILE=sn_result_${REV_NUM}.txt

echo  |tee -a $TOTAL_RESULT_FILE
echo "sn test" |tee -a $TOTAL_RESULT_FILE

cd $DIR_UNNORM/$TR_UN
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

cd $DIR_RENORM/$TR_RE
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

echo "####" |tee $RESULT_FILE
echo "1, /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE
echo "2, /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE

echo "####h5diff $TR_UN vs $TR_RE chk file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/chk0005.4d.hdf5 $DIR_RENORM/$TR_RE/chk0005.4d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0005.2d.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0005.2d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt map file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0005.2d.map.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0005.2d.map.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE

#######################################################################################################################################
# boltzmann drift test
TR_UN=drift_bol_trunk_unnorm
TR_RE=drift_bol_trunk_renorm
INPUT_FILE=kinetic.in
RESULT_FILE=drift_bol_result_${REV_NUM}.txt

echo  |tee -a $TOTAL_RESULT_FILE
echo "drift boltzman test" |tee -a $TOTAL_RESULT_FILE

cd $DIR_UNNORM/$TR_UN
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

cd $DIR_RENORM/$TR_RE
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

echo "####" |tee $RESULT_FILE
echo "1, /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE
echo "2, /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE

echo "####h5diff $TR_UN vs $TR_RE chk file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/chk0010.4d.hdf5 $DIR_RENORM/$TR_RE/chk0010.4d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0010.2d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt map file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0010.2d.map.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE


#######################################################################################################################################
# sn old test
TR_UN=snold_trunk_unnorm
TR_RE=snold_trunk_renorm
INPUT_FILE=sn_withE.in
RESULT_FILE=snold_result_${REV_NUM}.txt

echo  |tee -a $TOTAL_RESULT_FILE
echo "sn old test" |tee -a $TOTAL_RESULT_FILE

cd $DIR_UNNORM/$TR_UN
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

cd $DIR_RENORM/$TR_RE
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

echo "####" |tee $RESULT_FILE
echo "1, /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE
echo "2, /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE

echo "####h5diff $TR_UN vs $TR_RE chk file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/chk0005.4d.hdf5 $DIR_RENORM/$TR_RE/chk0005.4d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0005.2d.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0005.2d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt map file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0005.2d.map.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0005.2d.map.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE

#######################################################################################################################################
# sn older test
TR_UN=snolder_trunk_unnorm
TR_RE=snolder_trunk_renorm
INPUT_FILE=sn_withE.in
RESULT_FILE=snolder_result_${REV_NUM}.txt

echo  |tee -a $TOTAL_RESULT_FILE
echo "sn older test" |tee -a $TOTAL_RESULT_FILE

cd $DIR_UNNORM/$TR_UN
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

cd $DIR_RENORM/$TR_RE
source ./cleandata
mpirun -np 2 /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex $INPUT_FILE |tee output.txt
cd ../../

echo "####" |tee $RESULT_FILE
echo "1, /home/wonjae/ws/trunkESL/$RUN_PATH_UN/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE
echo "2, /home/wonjae/ws/trunkESL/$RUN_PATH_RE/exec/cogent.Linux.64.mpicxx.gfortran.OPT.MPI.ex" |tee -a $RESULT_FILE

echo "####h5diff $TR_UN vs $TR_RE chk file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/chk0005.4d.hdf5 $DIR_RENORM/$TR_RE/chk0005.4d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0005.2d.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0005.2d.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####h5diff $TR_UN vs $TR_RE plt map file" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
h5diff $DIR_UNNORM/$TR_UN/plt_density_plots/plt.1.hydrogen.density0005.2d.map.hdf5 $DIR_RENORM/$TR_RE/plt_density_plots/plt.1.hydrogen.density0005.2d.map.hdf5 |tee -a $RESULT_FILE $TOTAL_RESULT_FILE
echo "####" |tee -a $RESULT_FILE $TOTAL_RESULT_FILE


