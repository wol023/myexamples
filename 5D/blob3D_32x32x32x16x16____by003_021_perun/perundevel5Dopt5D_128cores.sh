#PBS -N cogent5d
#PBS -V
#PBS -l nodes=4:ppn=12

#cd /home/nfs/wol023/esl/myexamples/5D/blob3D_32x32x32x16x16____by003_021
cd $PBS_O_WORKDIR

mpirun ../../PERUNcogent5Dopt5D.ex kinetic.in >& perunoutput.out

