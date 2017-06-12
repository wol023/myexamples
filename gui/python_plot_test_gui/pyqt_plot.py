import sys
import os, fnmatch
from PyQt4 import QtCore, QtGui, uic

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1 
from numpy.ma import masked_array
import matplotlib.colors as mcolors

#for fft
import scipy
import scipy.fftpack

#To parse n0_grid_func
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy import *
from sympy.abc import x, y, z

#for least squre fit
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

# label formatter
import matplotlib.ticker as ticker 

# slider 3
from matplotlib.widgets import Slider

#to 3D plot
#from mayavi import mlab
#import mayavi

#alternative 3D plot
from mpl_toolkits.mplot3d import Axes3D





if os.path.isfile("plot_calc.ui"):
    qtCreatorFile = "plot_calc.ui" # Enter file here.
else:
    qtCreatorFile = "plot_calc_perun.ui" # Enter file here.
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.np_vpar_maxwell2D_diff_time=[]
        self.np_time=[]
        self.np_hf_time=[]
        self.np_hf_amp_time=[]
        self.np_lf_time=[]
        self.np_lf_amp_time=[]
        self.np_all_amp_time=[]
        self.np_all_time=[]
        self.start_time=0.0
        self.end_time=1.0

        self.colors=('k','y','m','c','b','g','r','#aaaaaa')
        self.linestyles=('-','--','-.',':')
        self.styles=[(color,linestyle) for linestyle in self.linestyles for color in self.colors]

        self.open_target_button.clicked.connect(self.print_target_file_lines)
        self.open_load_button.clicked.connect(self.print_current_dir_lines)
        self.button_plot_selected.clicked.connect(self.plotFigure)

    def check_and_try_cluster(self,selected_file,status=0):
        ###check cluster
        head=os.path.split((selected_file))
        path_loc=head[0]
        file_loc=head[1]

        homedir=os.getcwd()
        if not os.path.exists(path_loc):
            os.mkdir(path_loc)
            self.print_ui(path_loc+'/'+' is created.')

        status=0
        currentdirlist=os.listdir(path_loc)
        if file_loc in currentdirlist:
            self.print_ui(file_loc+' is found in local machine.')
            status = 1
        else:

            if (self.cb_remote.checkState()):
                self.print_ui(file_loc+' is NOT found in local machine.. start downloading.')
                # sftp
                import base64
                import pysftp
                cnopts = pysftp.CnOpts()
                cnopts.hostkeys = None
                # sftp

                #if self.tbox_host.toPlainText()!=[] and self.tbox_user.toPlainText()!=[] and self.tbox_pword.toPlainText()!=[] and self.tbox_basepath.toPlainText()!=[] and self.tbox_targetpath.toPlainText()!=[]:

                if self.tbox_host.toPlainText()!=[] and self.tbox_user.toPlainText()!=[] and self.le_password.text()!=[] and self.tbox_basepath.toPlainText()!=[] and self.tbox_targetpath.toPlainText()!=[]:
                    host=str(self.tbox_host.toPlainText())
                    username=str(self.tbox_user.toPlainText())
                    #password=str(self.tbox_pword.toPlainText())
                    password=str(self.le_password.text())
                    basepath=str(self.tbox_basepath.toPlainText())
                    targetpath=str(self.tbox_targetpath.toPlainText())

                    self.tbox_selected_download.setText(file_loc)
                    #with pysftp.Connection(host=host, username=username, password=base64.b64decode(password),cnopts=cnopts) as sftp:
                    with pysftp.Connection(host=host, username=username, password=password,cnopts=cnopts) as sftp:
                        if sftp.exists(basepath):
                            with sftp.cd(basepath):
                                if sftp.exists(targetpath):
                                    with sftp.cd(targetpath):
                                            if sftp.exists(path_loc):
                                                with sftp.cd(path_loc):
                                                    if sftp.exists(file_loc):
                                                        os.chdir(path_loc)
                                                        sftp.get(file_loc, preserve_mtime=True,callback=self.printProgress)
                                                        self.print_ui(file_loc+' download completed.')
                                                        os.chdir(homedir)
                                                        status=2
                                                    else:
                                                        self.print_ui(file_loc+' is not found in '+host)
                                                        status=-1
                                            else:
                                                self.print_ui(path_loc+' is not found in '+host)
                                                status=-2

                                else:
                                    self.print_ui(targetpath+' is not found in '+host)
                                    status=-3
                        else:
                            self.print_ui(basepath+' is not found in '+host)
                            status=-4
            else:
                self.print_ui('The file was not found on machine, download from remote disabled.')

        self.print_ui('file check is completed with status = '+str(status))
        return status
        ## check cluster

    def import_multdim_comps(self,filename,withghost=0):
        self.print_ui('import_multidim_comps()')
        try:
            filename
        except NameError:
            self.print_ui(filename+' is NOT found')
        else:
            #self.print_ui(filename+' is found')
            File =h5py.File(filename,'r')     
            #print File['Chombo_global'].attrs['SpaceDim']
            #print 
            #print File['level_0']['boxes'][:]
            #print 
            #print File['level_0']['data:offsets=0'][:]
            #print 
            #print len(File['level_0']['data:datatype=0'][:])
            #print 
            #print File['level_0']['data_attributes'].attrs.items()
            #print 
            #print File['level_0']['data_attributes'].attrs.keys()
            
            cogent_time = File['level_0'].attrs.get('time')
            self.print_ui('cogent_time ='+str(cogent_time))
    
            ghost = File['level_0']['data_attributes'].attrs.get('ghost')
            self.print_ui('ghost='+str(ghost))
            comps = File['level_0']['data_attributes'].attrs.get('comps')
            self.print_ui('comps='+str(comps))
        
            boxes=File['level_0']['boxes'][:]
            num_decomposition = len(boxes)#boxes.shape[0]
            self.print_ui('num_decomp='+str(num_decomposition))
            
            dim=len(boxes[0])/2
            self.print_ui('dim='+str(dim))
            
            min_box_intvect=np.ones(dim*2)
            max_box_intvect=np.ones(dim*2)
            min_box_intvect=min_box_intvect.astype(int)
            max_box_intvect=max_box_intvect.astype(int)
            
            total_box=np.zeros(( num_decomposition,dim*2 ))
            total_box=total_box.astype(int)
            
            for i in range(dim*2):
                for j in range(num_decomposition):
                    total_box[j][i]=boxes[j][i]
            #print total_box
            
            for i in range(dim*2):
                min_box_intvect[i]=min(total_box[:,i])
                max_box_intvect[i]=max(total_box[:,i])
            
            #print 'lo=',min_box_intvect
            #print 'hi=',max_box_intvect
            domain_box_intvect=min_box_intvect
            domain_box_intvect[dim:dim*2]=max_box_intvect[dim:dim*2]
            #print 'domain=',domain_box_intvect
            
            shifter=np.zeros(dim*2)
            shifter=shifter.astype(int)
            for i in range(dim):
                shifter[i]=-domain_box_intvect[i]
            for i in range(dim):
                shifter[i+dim]=-domain_box_intvect[i]
            #print 'domian_shifter=',shifter
        
            if dim<3:
                dir_x=0
                dir_y=1
            elif dim ==3:
                dir_x=0
                dir_y=1
                dir_z=2
            elif dim ==4:
                dir_x=0
                dir_y=1
                dir_vpar=2
                dir_mu=3
            elif dim ==5:
                dir_x=0
                dir_y=1
                dir_z=2
                dir_vpar=3
                dir_mu=4
     
            num_cell_loc=np.ones(dim)
            num_cell_loc=num_cell_loc.astype(int)
            for i in range(dim):
                xcell_beg=boxes[0][i]
                xcell_fin=boxes[0][i+dim]
                num_xcell=xcell_fin-xcell_beg+1
                #print xcell_beg,xcell_fin,num_xcell
                num_cell_loc[i]=num_xcell
            #print 'num_cell_loc=',num_cell_loc
            prod_num_cell_loc = np.prod(num_cell_loc)
            #print 'prod_num_cell_loc=',prod_num_cell_loc
            num_cell_loc_comps=np.append(num_cell_loc,[comps])
            #print 'num_cell_loc_comps=',num_cell_loc_comps
            prod_num_cell_loc_comps = np.prod(num_cell_loc_comps)
            #print 'prod_num_cell_loc_comps=',prod_num_cell_loc_comps
           
            num_cell_loc_with_ghost=np.ones(dim)
            num_cell_loc_with_ghost=num_cell_loc_with_ghost.astype(int)
            for i in range(dim):
                num_cell_loc_with_ghost[i]=num_cell_loc[i]+2*ghost[i]
            #print 'num_cell_loc_with_ghost=',num_cell_loc_with_ghost
            prod_num_cell_loc_with_ghost = np.prod(num_cell_loc_with_ghost)
            #print 'prod_num_cell_loc_with_ghost=',prod_num_cell_loc_with_ghost
            num_cell_loc_with_ghost_comps=np.append(num_cell_loc_with_ghost,[comps])
            #print 'num_cell_loc_with_ghost_comps=',num_cell_loc_with_ghost_comps
            prod_num_cell_loc_with_ghost_comps = np.prod(num_cell_loc_with_ghost_comps)
            #print 'prod_num_cell_loc_with_ghost_comps=',prod_num_cell_loc_with_ghost_comps
            num_cell_loc_with_ghost_tuple=()
            for i in range(len(num_cell_loc_with_ghost)):
                num_cell_loc_with_ghost_tuple=num_cell_loc_with_ghost_tuple+(num_cell_loc_with_ghost[i],)
            self.print_ui('num_cell_loc_with_ghost_tuple='+str(num_cell_loc_with_ghost_tuple))
    
    
    
         
            num_cell_total=np.ones(dim)
            num_cell_total=num_cell_total.astype(int)
            for i in range(dim):
                xcell_beg=min_box_intvect[i]
                xcell_fin=max_box_intvect[i+dim]
                num_xcell=xcell_fin-xcell_beg+1
                #print xcell_beg,xcell_fin,num_xcell
                num_cell_total[i]=num_xcell
            #print 'num_cell_total=',num_cell_total
            prod_num_cell_total = np.prod(num_cell_total)
            #print 'prod_num_cell_total=',prod_num_cell_total
        
            num_cell_total_comps=np.append(num_cell_total,[comps])
            #print 'num_cell_total_comps=',num_cell_total_comps
            prod_num_cell_total_comps = np.prod(num_cell_total_comps)
            #print 'prod_num_cell_total_comps=',prod_num_cell_total_comps
        
            decomposition=np.ones(dim)
            decomposition=decomposition.astype(int)
            for i in range(dim):
                decomposition[i] = num_cell_total[i]/num_cell_loc[i]
            #print 'decomposition=',decomposition
        
            num_cell_total_with_ghost=np.ones(dim)
            num_cell_total_with_ghost=num_cell_total_with_ghost.astype(int)
            for i in range(dim):
                num_cell_total_with_ghost[i]=(num_cell_loc[i]+2*ghost[i])*decomposition[i]
            #print 'num_cell_total_with_ghost=',num_cell_total_with_ghost
            prod_num_cell_total_with_ghost = np.prod(num_cell_total_with_ghost)
            #print 'prod_num_cell_total_with_ghost=',prod_num_cell_total_with_ghost
        
        
            num_cell_total_with_ghost_comps=np.append(num_cell_total_with_ghost,[comps])
            #print 'num_cell_total_with_ghost_comps=',num_cell_total_with_ghost_comps
            prod_num_cell_total_with_ghost_comps = np.prod(num_cell_total_with_ghost_comps)
            #print 'prod_num_cell_total_with_ghost_comps=',prod_num_cell_total_with_ghost_comps
        
        
            num_cell_total_with_outer_ghost=np.ones(dim)
            num_cell_total_with_outer_ghost=num_cell_total_with_outer_ghost.astype(int)
            for i in range(dim):
                num_cell_total_with_outer_ghost[i]=num_cell_loc[i]*decomposition[i]+2*ghost[i]
            #print 'num_cell_total_with_outer_ghost=',num_cell_total_with_outer_ghost
            prod_num_cell_total_with_outer_ghost = np.prod(num_cell_total_with_outer_ghost)
            #print 'prod_num_cell_total_with_outer_ghost=',prod_num_cell_total_with_outer_ghost
        
            num_cell_total_with_outer_ghost_comps=np.append(num_cell_total_with_outer_ghost,[comps])
            #print 'num_cell_total_with_outer_ghost_comps=',num_cell_total_with_outer_ghost_comps
            prod_num_cell_total_with_outer_ghost_comps = np.prod(num_cell_total_with_outer_ghost_comps)
            #print 'prod_num_cell_total_with_outer_ghost_comps=',prod_num_cell_total_with_outer_ghost_comps
        
            #import varialbe data
            data = File['level_0']['data:datatype=0'][:]
            previous_index=0
            
            dataNd_bvec_with_ghost_comps=np.linspace(0.0,0.1,num=prod_num_cell_total_with_ghost_comps).reshape((num_cell_total_with_ghost_comps))
            dataNd_bvec_comps=np.linspace(0.0,0.1,num=prod_num_cell_total_comps).reshape((num_cell_total_comps))
            dataNd_bvec_with_outer_ghost_comps=np.linspace(0.0,0.1,num=prod_num_cell_total_with_outer_ghost_comps).reshape((num_cell_total_with_outer_ghost_comps))
        
            dataNd_loc_with_ghost_comps=np.linspace(0.0,0.1,num=prod_num_cell_loc_with_ghost_comps).reshape((num_cell_loc_with_ghost_comps))
            
            cells_shift=np.zeros(dim*2)
            cells_shift=cells_shift.astype(int)
            cells_shift_with_ghost=np.zeros(dim*2)
            cells_shift_with_ghost=cells_shift_with_ghost.astype(int)

            self.pb_download.setValue(0)
         
            for i in range(num_decomposition):
                cells = File['level_0']['boxes'][i]
                #sys.stdout.write('.')
                #sys.stdout.flush()
                self.printProgress(i,num_decomposition-1)
                #print 'cells=',cells
                
                for j in range(len(cells_shift_with_ghost)):
                    cells_shift[j]=cells[j]+shifter[j]
                #print cells_shift
        
                for j in range(dim):
                    temp_decomp= cells_shift[j]/num_cell_loc[j]
                    cells_shift_with_ghost[j]=cells_shift[j]+ghost[j]*temp_decomp*2
                    cells_shift_with_ghost[j+dim]=(cells_shift[j]+ghost[j]*temp_decomp*2)+num_cell_loc_with_ghost[j]-1
        
                #print 'cells=',cells
                #print 'cells_shift=',cells_shift
                #print 'cells_shift_with_ghost=',cells_shift_with_ghost
                    
                for d in range(comps):
                    #print 'dim=',dim
                    #print 'i=',i
                    #print 'd=',d
                    #print 'previous_index=',previous_index
                    #print 'loc_end_index =',prod_num_cell_loc_with_ghost*(d+1)+prod_num_cell_loc_with_ghost*(i)*comps
                    if dim==3:
                        dataNd_loc_with_ghost_comps[:,:,:,d]=data[previous_index:prod_num_cell_loc_with_ghost*(d+1)+prod_num_cell_loc_with_ghost*(i)*comps].reshape((num_cell_loc_with_ghost[dir_x],num_cell_loc_with_ghost[dir_y],num_cell_loc_with_ghost[dir_z]),order='F') 
                    elif dim==5:
                        dataNd_loc_with_ghost_comps[:,:,:,:,:,d]=data[previous_index:prod_num_cell_loc_with_ghost*(d+1)+prod_num_cell_loc_with_ghost*(i)*comps].reshape(num_cell_loc_with_ghost_tuple,order='F') 
    
                    previous_index=prod_num_cell_loc_with_ghost*(d+1)+prod_num_cell_loc_with_ghost*(i)*comps
                        
            
                for d in range(comps):
                    if dim==3:
                        dataNd_bvec_with_ghost_comps[cells_shift_with_ghost[dir_x]:cells_shift_with_ghost[dim+dir_x]+1, cells_shift_with_ghost[dir_y]:cells_shift_with_ghost[dim+dir_y]+1, cells_shift_with_ghost[dir_z]:cells_shift_with_ghost[dim+dir_z]+1,d]=dataNd_loc_with_ghost_comps[:,:,:,d]
                    elif dim==5:
                        dataNd_bvec_with_ghost_comps[cells_shift_with_ghost[dir_x]:cells_shift_with_ghost[dim+dir_x]+1, cells_shift_with_ghost[dir_y]:cells_shift_with_ghost[dim+dir_y]+1, cells_shift_with_ghost[dir_z]:cells_shift_with_ghost[dim+dir_z]+1, cells_shift_with_ghost[dir_vpar]:cells_shift_with_ghost[dim+dir_vpar]+1,cells_shift_with_ghost[dir_mu]:cells_shift_with_ghost[dim+dir_mu]+1,d]=dataNd_loc_with_ghost_comps[:,:,:,:,:,d]
    
            
            
            File.close()
        
            #If we have nonzero ghost cells then remove them
            sum_ghosts=0
            for i in range(len(ghost)):
                sum_ghosts=sum_ghosts+ghost[i]
            if sum_ghosts>0: 
                #removing all ghost cells
                if dim==2:
                    for i in range(num_cell_total[0]):
                        current_decomp_i=i/num_cell_loc[0]
                        for j in range(num_cell_total[1]):
                            current_decomp_j=j/num_cell_loc[1]
                            for d in range(comps):
                                dataNd_bvec_comps[i,j,d]=dataNd_bvec_with_ghost_comps[ghost[0]+ghost[0]*2*current_decomp_i+i,ghost[1]+ghost[1]*2*current_decomp_j+j,d]
                elif dim==3:
                    for i in range(num_cell_total[0]):
                        current_decomp_i=i/num_cell_loc[0]
                        for j in range(num_cell_total[1]):
                            current_decomp_j=j/num_cell_loc[1]
                            for k in range(num_cell_total[2]):
                                current_decomp_k=k/num_cell_loc[2]
                                for d in range(comps):
                                    dataNd_bvec_comps[i,j,k,d]=dataNd_bvec_with_ghost_comps[ghost[0]+ghost[0]*2*current_decomp_i+i,ghost[1]+ghost[1]*2*current_decomp_j+j,ghost[2]+ghost[2]*2*current_decomp_k+k,d]
                elif dim==4:
                    for i in range(num_cell_total[0]): #x
                        current_decomp_i=i/num_cell_loc[0]
                        for j in range(num_cell_total[1]): #y
                            current_decomp_j=j/num_cell_loc[1]
                            for k in range(num_cell_total[2]): #vpar
                                current_decomp_k=k/num_cell_loc[2]
                                for l in range(num_cell_total[3]): #mu
                                    current_decomp_l=l/num_cell_loc[3]
                                    for d in range(comps): #components
                                        dataNd_bvec_comps[i,j,k,l,d]=dataNd_bvec_with_ghost_comps[ghost[0]+ghost[0]*2*current_decomp_i+i,ghost[1]+ghost[1]*2*current_decomp_j+j,ghost[2]+ghost[2]*2*current_decomp_k+k, ghost[3]+ghost[3]*2*current_decomp_l+l, d]
                elif dim==5:
                    for i in range(num_cell_total[0]): #x
                        current_decomp_i=i/num_cell_loc[0]
                        for j in range(num_cell_total[1]): #y
                            current_decomp_j=j/num_cell_loc[1]
                            for k in range(num_cell_total[2]): #z
                                current_decomp_k=k/num_cell_loc[2]
                                for l in range(num_cell_total[3]): #vpar
                                    current_decomp_l=l/num_cell_loc[3]
                                    for m in range(num_cell_total[4]): #mu
                                        current_decomp_m=m/num_cell_loc[4]
                                        for d in range(comps): #components
                                            dataNd_bvec_comps[i,j,k,l,m,d]=dataNd_bvec_with_ghost_comps[ghost[0]+ghost[0]*2*current_decomp_i+i,ghost[1]+ghost[1]*2*current_decomp_j+j,ghost[2]+ghost[2]*2*current_decomp_k+k, ghost[3]+ghost[3]*2*current_decomp_l+l, ghost[4]+ghost[4]*2*current_decomp_m+m,  d]
        
    
    
        
                #removing inner ghost cells
                if dim==3:
                    for i in range(num_cell_total_with_outer_ghost[0]):
                        current_decomp_i=(i+ghost[0])/num_cell_loc[0]
                        if current_decomp_i>decomposition[0]:
                            last_decomp_i=1
                        elif current_decomp_i==0:
                            last_decomp_i=-1
                        else:
                            last_decomp_i=0
                        for j in range(num_cell_total_with_outer_ghost[1]):
                            current_decomp_j=(j+ghost[1])/num_cell_loc[1]
                            if current_decomp_j>decomposition[1]:
                                last_decomp_j=1
                            elif current_decomp_j==0:
                                last_decomp_j=-1
                            else:
                                last_decomp_j=0
                            for k in range(num_cell_total_with_outer_ghost[2]):
                                current_decomp_k=(k+ghost[2])/num_cell_loc[2]
                                if current_decomp_k>decomposition[2]:
                                    last_decomp_k=1
                                elif current_decomp_k==0:
                                    last_decomp_k=-1
                                else:
                                    last_decomp_k=0
                                for d in range(comps):
                                    dataNd_bvec_with_outer_ghost_comps[i,j,k,d]=dataNd_bvec_with_ghost_comps[ghost[0]*2*(current_decomp_i-last_decomp_i-1)+i,ghost[1]*2*(current_decomp_j-last_decomp_j-1)+j,ghost[2]*2*(current_decomp_k-last_decomp_k-1)+k,d]
                elif dim==4:
                    for i in range(num_cell_total_with_outer_ghost[0]): #x
                        current_decomp_i=(i+ghost[0])/num_cell_loc[0]
                        if current_decomp_i>decomposition[0]:
                            last_decomp_i=1
                        elif current_decomp_i==0:
                            last_decomp_i=-1
                        else:
                            last_decomp_i=0
                        for j in range(num_cell_total_with_outer_ghost[1]): #y
                            current_decomp_j=(j+ghost[1])/num_cell_loc[1]
                            if current_decomp_j>decomposition[1]:
                                last_decomp_j=1
                            elif current_decomp_j==0:
                                last_decomp_j=-1
                            else:
                                last_decomp_j=0
                            for k in range(num_cell_total_with_outer_ghost[2]): #vpar
                                current_decomp_k=(k+ghost[2])/num_cell_loc[2]
                                if current_decomp_k>decomposition[2]:
                                    last_decomp_k=1
                                elif current_decomp_k==0:
                                    last_decomp_k=-1
                                else:
                                    last_decomp_k=0
                                for l in range(num_cell_total_with_outer_ghost[3]): #mu
                                    current_decomp_l=(l+ghost[3])/num_cell_loc[3]
                                    if current_decomp_l>decomposition[3]:
                                        last_decomp_l=1
                                    elif current_decomp_l==0:
                                        last_decomp_l=-1
                                    else:
                                        last_decomp_l=0
                                    for d in range(comps):
                                        dataNd_bvec_with_outer_ghost_comps[i,j,k,l,d]=dataNd_bvec_with_ghost_comps[ghost[0]*2*(current_decomp_i-last_decomp_i-1)+i,ghost[1]*2*(current_decomp_j-last_decomp_j-1)+j,ghost[2]*2*(current_decomp_k-last_decomp_k-1)+k,ghost[3]*2*(current_decomp_l-last_decomp_l-1)+l,d]
                elif dim==5:
                    for i in range(num_cell_total_with_outer_ghost[0]): #x
                        current_decomp_i=(i+ghost[0])/num_cell_loc[0]
                        if current_decomp_i>decomposition[0]:
                            last_decomp_i=1
                        elif current_decomp_i==0:
                            last_decomp_i=-1
                        else:
                            last_decomp_i=0
                        for j in range(num_cell_total_with_outer_ghost[1]): #y
                            current_decomp_j=(j+ghost[1])/num_cell_loc[1]
                            if current_decomp_j>decomposition[1]:
                                last_decomp_j=1
                            elif current_decomp_j==0:
                                last_decomp_j=-1
                            else:
                                last_decomp_j=0
                            for k in range(num_cell_total_with_outer_ghost[2]): #z
                                current_decomp_k=(k+ghost[2])/num_cell_loc[2]
                                if current_decomp_k>decomposition[2]:
                                    last_decomp_k=1
                                elif current_decomp_k==0:
                                    last_decomp_k=-1
                                else:
                                    last_decomp_k=0
                                for l in range(num_cell_total_with_outer_ghost[3]): #vpar
                                    current_decomp_l=(l+ghost[3])/num_cell_loc[3]
                                    if current_decomp_l>decomposition[3]:
                                        last_decomp_l=1
                                    elif current_decomp_l==0:
                                        last_decomp_l=-1
                                    else:
                                        last_decomp_l=0
                                    for m in range(num_cell_total_with_outer_ghost[4]): #mu
                                        current_decomp_m=(m+ghost[4])/num_cell_loc[4]
                                        if current_decomp_m>decomposition[4]:
                                            last_decomp_m=1
                                        elif current_decomp_m==0:
                                            last_decomp_m=-1
                                        else:
                                            last_decomp_m=0
                                        for d in range(comps):
                                            dataNd_bvec_with_outer_ghost_comps[i,j,k,l,m,d]=dataNd_bvec_with_ghost_comps[ghost[0]*2*(current_decomp_i-last_decomp_i-1)+i,ghost[1]*2*(current_decomp_j-last_decomp_j-1)+j,ghost[2]*2*(current_decomp_k-last_decomp_k-1)+k,ghost[3]*2*(current_decomp_l-last_decomp_l-1)+l,ghost[4]*2*(current_decomp_m-last_decomp_m-1)+m,d]
    
            else:
                #just copy
                if dim==2:
                    dataNd_bvec_comps[:,:,:]                       =dataNd_bvec_with_ghost_comps[:,:,:]
                    dataNd_bvec_with_outer_ghost_comps[:,:,:]      =dataNd_bvec_with_ghost_comps[:,:,:]
                elif dim==3:
                    dataNd_bvec_comps[:,:,:,:]                     =dataNd_bvec_with_ghost_comps[:,:,:,:]
                    dataNd_bvec_with_outer_ghost_comps[:,:,:,:]    =dataNd_bvec_with_ghost_comps[:,:,:,:]
                elif dim==4:
                    dataNd_bvec_comps[:,:,:,:,:]                   =dataNd_bvec_with_ghost_comps[:,:,:,:,:]
                    dataNd_bvec_with_outer_ghost_comps[:,:,:,:,:]  =dataNd_bvec_with_ghost_comps[:,:,:,:,:]
                elif dim==5:
                    dataNd_bvec_comps[:,:,:,:,:,:]                 =dataNd_bvec_with_ghost_comps[:,:,:,:,:,:]
                    dataNd_bvec_with_outer_ghost_comps[:,:,:,:,:,:]=dataNd_bvec_with_ghost_comps[:,:,:,:,:,:]
    
            
            self.tbox_selected_download.setText(filename)
            #print ' collected ', num_decomposition, 'decompositions'
            self.print_ui(' collected '+str(num_decomposition)+' decompositions')
            if (withghost==1):
                self.print_ui('Added outer ghost cells'+str(ghost))
                self.print_ui(str(num_cell_total_comps)+'->'+str(dataNd_bvec_with_outer_ghost_comps.shape))
                return dataNd_bvec_with_outer_ghost_comps, ghost, cogent_time
            elif (withghost==2):
                self.print_ui('Added inner and outer ghost cells'+str(ghost))
                self.print_ui(str(num_cell_total_comps)+'->'+str(dataNd_bvec_with_ghost_comps.shape))
                return dataNd_bvec_with_ghost_comps, ghost, cogent_time
            else:
                self.print_ui(str(num_cell_total_comps)+'->'+str(dataNd_bvec_comps.shape))
                return dataNd_bvec_comps, ghost, cogent_time

    def printProgress(self, transferred, toBeTransferred):
        #print "Transferred: {:5.2f} %\r".format(float(transferred)/toBeTransferred*100)
        #self.print_ui("Transferred: {:5.2f} %".format(float(transferred)/toBeTransferred*100))
        self.pb_download.setValue(float(transferred)/toBeTransferred*100)
        QtGui.QApplication.processEvents()

    def print_ui(self,stream):
        self.results_window.append(stream)
        QtGui.QApplication.processEvents()

    def print_current_dir_lines(self):
        self.lw_target.clear()
        current_dir = str(self.current_dir_box.toPlainText())
        self.print_ui(current_dir)
        hdf5files=[]
        for root, subFolders, files in os.walk(current_dir):
            for a_file in files:
                if a_file.endswith('.hdf5') and not a_file.endswith('.map.hdf5') and a_file.startswith('plt.'):
                    hdf5files.append(os.path.join(root,a_file))
                    #self.print_ui(a_file)
        for item in hdf5files:
            self.lw_target.addItem(item)

        #sort items
        #self.lw_target.sortItems(QtCore.Qt.DescendingOrder)
        self.lw_target.sortItems()

        ## make output directory
        plot_output='./python_auto_plots.0.dir'
        self.tbox_localout.clear()
        notunique=1
        while notunique:
            if os.path.exists(plot_output):
                notunique=1
                int_oldnum=int(plot_output.split('./python_auto_plots.')[1].split('.dir')[0])
                str_newnum=str(int_oldnum+1)
                plot_output='./python_auto_plots.'+str_newnum+'.dir'
            else:
                notunique=0
        self.tbox_localout.append(plot_output)
        # find and read input deck
        self.ref_time=self.read_input_deck()
        self.tbox_transit_time.setText(str(self.ref_time))

    def print_target_file_lines(self):
        self.lw_target.clear()
        target_file = (self.target_list_box.toPlainText())
        with open(target_file,'r') as fo:
            self.print_ui('target file opened.. ready.')
            lines= fo.read().splitlines()
            host=lines[0]
            user=lines[1]
            pword=lines[2]
            basepath=lines[3]
            targetpath=lines[4]
            nextline=lines[5]
            rawpathfiles=lines[6:]
        self.tbox_host.setText(host)
        self.tbox_user.setText(user)
        #self.tbox_pword.setText(pword)
        self.le_password.setText(pword)
        self.tbox_basepath.setText(basepath)
        self.tbox_targetpath.setText(targetpath)
        pathfiles=[]
        for i in range(len(rawpathfiles)):
            if rawpathfiles[i].lstrip().startswith('#'): #skip comment
                continue
            line =rawpathfiles[i].rstrip() #skip blank line
            if not line:
                continue 
            else:
                pathfiles.append(line)
        paths=[]
        files=[]
        for i in range(len(pathfiles)):
            head=os.path.split(pathfiles[i])
            paths.append(head[0])
            files.append(head[1])
        for i in range(len(pathfiles)):
            self.lw_target.addItem(paths[i]+'/'+files[i])
            if i==0:
                self.lw_target.setCurrentRow(0)
        #set first row
        ## make output directory
        plot_output='./python_auto_plots.0.dir'
        self.tbox_localout.clear()
        notunique=1
        while notunique:
            if os.path.exists(plot_output):
                notunique=1
                int_oldnum=int(plot_output.split('./python_auto_plots.')[1].split('.dir')[0])
                str_newnum=str(int_oldnum+1)
                plot_output='./python_auto_plots.'+str_newnum+'.dir'
            else:
                notunique=0
        self.tbox_localout.append(plot_output)
        # find and read input deck
        self.ref_time=self.read_input_deck()
        self.tbox_transit_time.setText(str(self.ref_time))

    def plotFigure(self):

        #initialize
        self.np_time=[]
        self.np_hf_time=[]
        self.np_hf_amp_time=[]
        self.np_lf_time=[]
        self.np_lf_amp_time=[]
        self.np_all_amp_time=[]
        self.np_all_time=[]

        selected_items=self.lw_target.selectedItems()
        selected_files=[]
        for i in range(len(selected_items)):
            selected_files.append(str(selected_items[i].text()))


        # initialize time dependent variables
        row_size=len(selected_files)
        col_size=int(self.te_z_total.toPlainText())/2+1


        for i in range(len(selected_files)):
            self.print_ui(selected_files[i])

            status=0
            status=self.check_and_try_cluster(selected_files[i])

            if status>0:
                x_pt=int(self.te_x_pt.toPlainText())-1
                y_pt=int(self.te_y_pt.toPlainText())-1
                z_pt=int(self.te_z_pt.toPlainText())-1
                vpar_pt=int(self.te_vpar_pt.toPlainText())-1
                mu_pt=int(self.te_mu_pt.toPlainText())-1
                plot_output=self.tbox_localout.toPlainText()
                if 'dfn' in selected_files[i]:
                    if 'hydrogen' in selected_files[i]:
                        speciesname='i'
                    elif 'electron' in selected_files[i]:
                        speciesname='e'
                    else:
                        speciesname='s'
                    #from plot_cogent_pack import plot_dfn
                    self.plot_dfn(selected_files[i],speciesname=speciesname,x_pt=x_pt,y_pt=y_pt,z_pt=z_pt,vpar_pt=vpar_pt,mu_pt=mu_pt,targetdir=plot_output)
                    if i==len(selected_files)-1:
                        if self.cb_vpar_maxwell2D_diff_time.checkState():
                            #print self.np_vpar_maxwell2D_diff_time
                            #print self.np_vpar_maxwell2D_diff_time.shape
                            self.plot_dfn_diff_time(selected_files[i],self.np_vpar_maxwell2D_diff_time,interpolation='none',title=r'$\log_{10} (|\delta f|)$',xlabel=r'$\bar{v}_\parallel$',ylabel='time ('+r'$\mu$'+'s)',targetdir=plot_output,symmetric_cbar=-1,cogent_time_start=self.start_time,cogent_time_end=self.end_time)


                if 'potential' in selected_files[i]:
                    self.plot_potential(selected_files[i],ghost=0,x_slice=x_pt,y_slice=x_pt,z_slice=z_pt,targetdir=plot_output)
                    if i==len(selected_files)-1:
                        head=os.path.split(selected_files[i])
                        path=head[0]
                        filename=head[1]
                        basedir=os.getcwd()
                        if self.cb_potential_fft_along_z.checkState():
                            #print 'hf:', self.np_hf_time
                            #print 'ha:', self.np_hf_amp_time
                            #print 'lf:',self.np_lf_time
                            #print 'la:', self.np_lf_amp_time

                            print 'time',self.np_time
                            print 'allf',self.np_all_time
                            print 'allA',self.np_all_amp_time
                            
                            # print file
                            if not os.path.exists(plot_output):
                                os.mkdir(plot_output)
                            os.chdir(plot_output)
                            with open(filename.replace('.hdf5','.potential_fft_all_time.dat'), 'w+') as fh:
                                #head line
                                buf = 'time' 
                                ind_c = 0 
                                while ind_c<col_size:
                                    buf=buf+'\tm=%d' % int(self.np_all_time[0,ind_c]/self.np_all_time[0,1])
                                    ind_c=ind_c+1
                                buf=buf+'\r\n'
                                fh.write(buf)
                                #data
                                ind_r = 0 
                                while ind_r<len(self.np_time):
                                    buf = '%g' % self.np_time[ind_r]
                                    ind_c=0
                                    while ind_c<col_size:
                                        buf = buf+'\t%g' % self.np_all_amp_time[ind_r,ind_c]
                                        ind_c=ind_c+1
                                    buf = buf+'\r\n'
                                    ind_r=ind_r+1
                                    fh.write(buf)
                            os.chdir(basedir)


                            plot_fft_amp=self.oplot_1d(var=np.log10(self.np_hf_amp_time),xaxis=self.np_time,title='fft',linewidth=1.5, linestyle='--',color='r',label='hf',xlabel='time',ylabel='log10(amplitude)')
                            plot_fft_amp=self.oplot_1d(var=np.log10(self.np_lf_amp_time),fig=plot_fft_amp,xaxis=self.np_time,title='fft',linewidth=1.5, linestyle='--',color='g',label='lf',xlabel='time',ylabel='log10(amplitude)')
                            if not os.path.exists(plot_output):
                                os.mkdir(plot_output)
                            os.chdir(plot_output)
                            plt.savefig(filename.replace('.hdf5','.potential_fft_amplitude_time.png'))
                            plt.savefig(filename.replace('.hdf5','.potential_fft_amplitude_time.eps'))
                            os.chdir(basedir)
                            plt.close(plot_fft_amp)

                            plot_fft=self.oplot_1d(var=self.np_hf_time,xaxis=self.np_time,title='fft',linewidth=1.5, linestyle='--',color='r',label='hf',xlabel='time',ylabel='wave number')
                            plot_fft=self.oplot_1d(var=self.np_lf_time,fig=plot_fft,xaxis=self.np_time,title='fft',linewidth=1.5, linestyle='--',color='g',label='lf',xlabel='time',ylabel='wave number')

                            if not os.path.exists(plot_output):
                                os.mkdir(plot_output)
                            os.chdir(plot_output)
                            plt.savefig(filename.replace('.hdf5','.potential_fft_frequency_time.png'))
                            plt.savefig(filename.replace('.hdf5','.potential_fft_frequency_time.eps'))
                            os.chdir(basedir)
                            plt.close(plot_fft)

                            #try plot all modes
                            ind =0
                            plot_fft_all=self.oplot_1d(var=np.log10(self.np_all_amp_time[:,0]),xaxis=self.np_time,title='fft',linewidth=1.5, linestyle=self.styles[ind][1],color=self.styles[ind][0],label=str(self.np_all_time[0,ind]/self.np_all_time[0,1])  ,xlabel='time',ylabel='log10(amplitude)')

                            ind=1
                            while (ind<int(self.te_z_total.toPlainText())/2+1):
                                plot_fft_all=self.oplot_1d(var=np.log10(self.np_all_amp_time[:,ind]),fig=plot_fft_all,xaxis=self.np_time,title='fft',linewidth=1.5, linestyle=self.styles[ind][1],color=self.styles[ind][0],label=str(self.np_all_time[0,ind]/self.np_all_time[0,1]),xlabel='time',ylabel='log10(amplitude)')
                                ind=ind+1

                            plt.legend(loc='best')

                            if not os.path.exists(plot_output):
                                os.mkdir(plot_output)
                            os.chdir(plot_output)
                            plt.savefig(filename.replace('.hdf5','.potential_fft_all_time.png'))
                            plt.savefig(filename.replace('.hdf5','.potential_fft_all_time.eps'))
                            os.chdir(basedir)
                            plt.close(plot_fft_all)



                if 'BField' in selected_files[i]:
                    plot_bvec(selected_files[i],ghost=0,targetdir=plot_output)
                if 'efield' in selected_files[i]:
                    plot_evec(selected_files[i],ghost=0,targetdir=plot_output)
                if 'density' in selected_files[i]:
                    if 'hydrogen' in selected_files[i]:
                        speciesname='i'
                    elif 'electron' in selected_files[i]:
                        speciesname='e'
                    else:
                        speciesname='s'
                    plot_density(selected_files[i],ghost=0,speciesname=speciesname,targetdir=plot_output)
            else:
                self.print_ui(selected_files[i]+ ' was NOT found.. skipping the file.')
        self.print_ui('ready')

    def read_input_deck(self,inputfile=''):
        if (inputfile==''):
            fname=self.find('*.in', './')
            inputfile=fname[0]
            self.tbox_inputfile.setText(inputfile)
        with open(inputfile,'r') as f:
            for line in f:
                #local vars
                units_temperature = 1
                units_mass = 1
                units_length = 1

                if line.lstrip().startswith('#'): #skip comment
                    continue
                line =line.rstrip() #skip blank line
                if not line:
                    continue 
                else: #noncomment line
                    strippedline=line
                    lhsrhs = strippedline.split("=")
                    l=0
                    while l<len(lhsrhs): #strip white spaces in lhs
                        lhsrhs[l]=lhsrhs[l].rstrip()
                        lhsrhs[l]=lhsrhs[l].lstrip()
                        l=l+1

                    if 'units.temperature' in lhsrhs[0]:
                        units_temperature = float(lhsrhs[1])
                        self.print_ui('IN:units_temperature = '+str(units_temperature))
                    if 'units.mass' in lhsrhs[0]:
                        units_mass = float(lhsrhs[1])
                        self.print_ui('IN:units_mass = '+str(units_mass))
                    if 'units.length' in lhsrhs[0]:
                        units_length = float(lhsrhs[1])
                        self.print_ui('IN:units_length= '+str(units_length))

                    
                    if '.x_max' in lhsrhs[0]:
                        self.te_x_length.setText(str(float(lhsrhs[1])))
                    if '.y_max' in lhsrhs[0]:
                        self.te_y_length.setText(str(float(lhsrhs[1])))
                    if '.z_max' in lhsrhs[0]:
                        self.te_z_length.setText(str(float(lhsrhs[1])))

                        
                    if '.history_indices' in lhsrhs[0]:
                        history_indices=lhsrhs[1].split()
                        if len(history_indices)==2:
                            self.te_x_pt.setText(str(history_indices[0]))
                            self.te_y_pt.setText(str(history_indices[1]))
                        if len(history_indices)==3:
                            self.te_x_pt.setText(str(history_indices[0]))
                            self.te_y_pt.setText(str(history_indices[1]))
                            self.te_z_pt.setText(str(history_indices[2]))
                    if 'gksystem.num_cells' in lhsrhs[0]:
                        num_cells=lhsrhs[1].split()
                        if len(num_cells)==4:
                            self.gksystem_dimension=4
                            self.te_x_total.setText(str(num_cells[0]))
                            self.te_y_total.setText(str(num_cells[1]))
                            self.te_vpar_total.setText(str(num_cells[2]))
                            self.te_mu_total.setText(str(num_cells[3]))
                        if len(num_cells)==5:
                            self.gksystem_dimension=5
                            self.te_x_total.setText(str(num_cells[0]))
                            self.te_y_total.setText(str(num_cells[1]))
                            self.te_z_total.setText(str(num_cells[2]))
                            self.te_vpar_total.setText(str(num_cells[3]))
                            self.te_mu_total.setText(str(num_cells[4]))
                    if 'gksystem.fixed_plot_indices' in lhsrhs[0]:
                        num_cells=lhsrhs[1].split()
                        if len(num_cells)==4:
                            self.te_x_fixed_plot.setText(str(num_cells[0]))
                            self.te_y_fixed_plot.setText(str(num_cells[1]))
                            self.te_vpar_fixed_plot.setText(str(num_cells[2]))
                            self.te_mu_fixed_plot.setText(str(num_cells[3]))
                        if len(num_cells)==5:
                            self.te_x_fixed_plot.setText(str(num_cells[0]))
                            self.te_y_fixed_plot.setText(str(num_cells[1]))
                            self.te_z_fixed_plot.setText(str(num_cells[2]))
                            self.te_vpar_fixed_plot.setText(str(num_cells[3]))
                            self.te_mu_fixed_plot.setText(str(num_cells[4]))
                            
            #end for
            self.te_x_length.setText(str( float(self.te_x_length.toPlainText())*units_length))
            self.te_y_length.setText(str( float(self.te_y_length.toPlainText())*units_length))
            self.te_z_length.setText(str( float(self.te_z_length.toPlainText())*units_length))

            self.print_ui('IN:x,y,z [m]= '+str(self.te_x_length.toPlainText())+str(self.te_y_length.toPlainText())+str(self.te_y_length.toPlainText()) )



        # calc ref_time from input deck parameters
        const_ELEMENTARY_CHARGE   = 1.60217653e-19 # C
        const_MASS_OF_PROTON      = 1.67262171e-27 # kg
        tempJoules = const_ELEMENTARY_CHARGE* units_temperature
        masskg = units_mass * const_MASS_OF_PROTON
        ref_speed = np.sqrt(tempJoules/masskg) 
        ref_time=units_length/ref_speed
        return ref_time

    def plot_dfn_diff_time(self,pathfilename,var,ghostIn=[],title='',xlabel='xlabel',ylabel='ylabel',xaxis=[],wh=1,fig_size_x=800,fig_size_y=600,sliced=0,x_slice=-1,y_slice=-1,z_slice=-1,label='',saveplots=1,showplots=0,ghost=0,targetdir=[],symmetric_cbar=0,interpolation='none',cogent_time_start=0.0, cogent_time_end=1.0):
        head=os.path.split(pathfilename)
        path=head[0]
        filename=head[1]
        basedir=os.getcwd()
        if targetdir==[]:
            targetdir='./python_auto_plots'

        self.init_plotting()
        fig_dfn_diff_time=plt.figure()
        plt.subplot(111)

        #plt.gca().margins(0.1, 0.1)
        if symmetric_cbar>0:
            v_min = self.var[:,:,0].min()
            v_max = var[:,:,0].max()
            if v_min<0 and v_max>0:
                v_min=-max(abs(v_min),abs(v_max))
                v_max=max(abs(v_min),abs(v_max))
                v_max=max(abs(v_min),abs(v_max))
                im=plt.imshow(var[:,:],vmin=v_min,vmax=v_max,interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
            else:
                im=plt.imshow(var[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
            self.add_colorbar(im,field=var[:,:])
        elif symmetric_cbar<0:
            #varp=var.copy()
            #varn=var.copy()
            #varp[np.where(var<=0)]=np.nan
            #varn[np.where(var>0)]=np.nan

            varp = masked_array(var,var<=0)
            varn = masked_array(var,var>=0)
            varlogp = varp.copy()
            varlogn = varn.copy()
            varlogp = np.log10(varlogp)
            varlogn = np.log10(-varlogn)

            varlogp = masked_array(varlogp,varlogp<-10)
            varlogn = masked_array(varlogn,varlogn<-10)

            #colors1 = plt.cm.Blues_r(np.linspace(0., 1, 128))
            #colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
            #colors = np.vstack((colors1, colors2))
            #mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

            #im = plt.imshow(var[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0,cmap=mymap)
            #plt.colorbar()


            #imp=plt.imshow(varp[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0,cmap=plt.get_cmap('Reds'))#float(num_ycell)/float(num_xcell))

            if self.ref_time!=-1.0:
                imp=plt.imshow(varlogp[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,cogent_time_start*self.ref_time*1.0E6,cogent_time_end*self.ref_time*1.0E6],aspect=2.0/((cogent_time_end-cogent_time_start)*self.ref_time*1.0E6) ,cmap=plt.get_cmap('Reds'))#float(num_ycell)/float(num_xcell))
            else:
                imp=plt.imshow(varlogp[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,cogent_time_start,cogent_time_end],aspect=2.0/((cogent_time_end-cogent_time_start)) ,cmap=plt.get_cmap('Reds'))#float(num_ycell)/float(num_xcell))
            cbp = plt.colorbar(imp,shrink=0.5)
            tick_locator = ticker.MaxNLocator(nbins=6)
            cbp.locator = tick_locator
            cbp.update_ticks()
            cbp.ax.set_title(r'     $\delta f>0$')
            #imn=plt.imshow(varn[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0,cmap=plt.get_cmap('Blues_r'))#float(num_ycell)/float(num_xcell))
            if self.ref_time!=-1.0:
                imn=plt.imshow(varlogn[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,cogent_time_start*self.ref_time*1.0E6,cogent_time_end*self.ref_time*1.0E6],aspect=2.0/((cogent_time_end-cogent_time_start)*self.ref_time*1.0E6) ,cmap=plt.get_cmap('Blues'))#float(num_ycell)/float(num_xcell))
            else:
                imn=plt.imshow(varlogn[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,cogent_time_start,cogent_time_end],aspect=2.0/((cogent_time_end-cogent_time_start)) ,cmap=plt.get_cmap('Blues'))#float(num_ycell)/float(num_xcell))
            cbn = plt.colorbar(imn,shrink=0.5)
            tick_locator = ticker.MaxNLocator(nbins=6)
            cbn.locator = tick_locator
            cbn.update_ticks()
            cbn.ax.set_title(r'     $\delta f<0$')
        else:
            im=plt.imshow(var[:,:],interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
            self.add_colorbar(im,field=var[:,:])
        plt.title(title)
        if xlabel=='xlabel':
            plt.xlabel(r'$\bar{v}_\parallel$')
        else:
            plt.xlabel(xlabel)
        if ylabel=='ylabel':
            plt.ylabel(r'$\bar{\mu}$')
        else:
            plt.ylabel(ylabel)
        fig_dfn_diff_time.set_tight_layout(True)
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
            os.chdir(targetdir)
            plt.savefig(filename.replace('.hdf5','.dfn_diff_time.png'))
            plt.savefig(filename.replace('.hdf5','.dfn_diff_time.eps'))
            os.chdir(basedir)
        if showplots==0:
            plt.close(fig_dfn_diff_time)

    
    def plot_potential(self,pathfilename,saveplots=1,ghost=0,x_slice=0.5,y_slice=0.5,z_slice=0.5,targetdir=[]):
        head=os.path.split(pathfilename)
        path=head[0]
        filename=head[1]
        #print path
        #print filename
        basedir=os.getcwd()
        if targetdir==[]:
            targetdir='./python_auto_plots'
        
        dataNd_potential_comps, num_ghost_potential, cogent_time=self.import_multdim_comps(filename=pathfilename)
        if ghost>0:
            dataNd_potential_with_outer_ghost_comps,num_ghost_potential,cogent_time=self.import_multdim_comps(filename=pathfilename,withghost=1)
        title_var='potential'
     ### read computational unit time
 
        if self.ref_time!=-1.0:
            title_var=title_var+'\n(t=%.2fE-6 s)'%(cogent_time*self.ref_time*1.0E6)
        else:
            title_var=title_var+'\n(t/t0=%.2f)'%cogent_time

    
        
        #fig=plot_Nd(dataNd_potential_comps,title=title_var)
        #if saveplots>0:
        #    if not os.path.exists(targetdir):
        #        os.mkdir(targetdir)
        #        print targetdir+'/', 'is created.'
        #    os.chdir(targetdir)
        #    plt.savefig(filename.replace('.hdf5','.potential_iso.png'))
        #    plt.savefig(filename.replace('.hdf5','.potential_iso.eps'))
        #    os.chdir(basedir)
        #if showplots==0:
        #    plt.close(fig)

        fig_potential=self.plot_Nd(dataNd_potential_comps,title=title_var,sliced=1,x_slice=x_slice,y_slice=y_slice,z_slice=z_slice)
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
            os.chdir(targetdir)

            if not os.path.exists('potential_slice'):
                os.mkdir('potential_slice')
            os.chdir('potential_slice')
            plt.savefig(filename.replace('.hdf5','.potential_slice.png'))
            plt.savefig(filename.replace('.hdf5','.potential_slice.eps'))
            os.chdir(basedir)
        if self.cb_3d_interactive.checkState()==0:
            plt.close(fig_potential)
    
        if ghost>0:
            #fig=plot_Nd(dataNd_potential_with_outer_ghost_comps,num_ghost_potential,title=title_var)
            #if saveplots>0:
            #    if not os.path.exists(targetdir):
            #        os.mkdir(targetdir)
            #    os.chdir(targetdir)
            #    plt.savefig(filename.replace('.hdf5','.potential_ghost_iso.png'))
            #    plt.savefig(filename.replace('.hdf5','.potential_ghost_iso.eps'))
            #    os.chdir(basedir)
            #if showplots==0:
            #    plt.close(fig)

            fig_potential=self.plot_Nd(dataNd_potential_with_outer_ghost_comps,num_ghost_potential,title=title_var,x_slice=x_slice,y_slice=y_slice,z_slice=z_slice)
            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                os.chdir(targetdir)
                plt.savefig(filename.replace('.hdf5','.potential_ghost_slice.png'))
                plt.savefig(filename.replace('.hdf5','.potential_ghost_slice.eps'))
                os.chdir(basedir)
            if self.cb_3d_interactive.checkState()==0:
                plt.close(fig_potential)

        
        if self.cb_potential_fft_along_z.checkState():
            nx = int(self.te_z_total.toPlainText())
            dx = 2.0*np.pi/nx;
            Lz = float(self.te_z_length.toPlainText())

            y = dataNd_potential_comps[x_slice,y_slice,:].flatten()
            left_x_cell= (0.0+dx/2.0)*Lz/(2.0*np.pi)
	    right_x_cell = (2.0*np.pi - dx/2.0)*Lz/(2.0*np.pi)
	    dimensional_x = np.linspace( left_x_cell, right_x_cell , nx)
	    x=dimensional_x/2/np.pi
	    
	    #T = (x[-1]-x[0])/nx
	    T = (x[-1]-x[0])/(nx-1)
	    xf = scipy.fftpack.fftfreq(nx,T)
	    yf = scipy.fftpack.fft(y)
	    xf = scipy.fftpack.fftshift(xf)
	    yf = scipy.fftpack.fftshift(yf)

	  
            xf_cp=xf.copy()
	    yf_cp=yf.copy()

            yf_cp = np.abs(yf_cp)

	    # flip copy
	    for idx, val in enumerate(xf_cp):
	        if val==0:
	            mid=idx
	    
	    if abs(xf_cp[0])>abs(xf_cp[-1]):
	        #print 'copy to left'
	        #print mid+1,idx,'->',0,mid-1
	        #print idx-(mid+1), mid-1-0
	        s1=mid+1
	        s2=idx
	        flag=-1 #to left
	    else:
	        #print 'copy to right'
	        #print 0,mid-1,'->',mid+1,idx
	        #print mid-1-0, idx-(mid+1)
	        s1=0
	        s2=mid-1
	        flag=1 #to right
	    
	    
	    for i in range(1,s2-s1+2):
                #print i, mid+i*flag, mid-i*flag
	        yf_cp[mid+i*flag]+=yf_cp[mid-i*flag]
	        yf_cp[mid-i*flag]=0


            if flag<0:
                if len(self.np_all_time)==0:
                    self.np_all_time=abs(xf_cp[0:mid+1][::-1])
                    self.np_all_amp_time=abs(yf_cp[0:mid+1][::-1])
                else:
                    self.np_all_time=np.vstack([self.np_all_time,abs(xf_cp[0:mid+1][::-1])])
                    self.np_all_amp_time=np.vstack([self.np_all_amp_time,abs(yf_cp[0:mid+1][::-1])])
            else:
                if len(self.np_all_time)==0:
                    self.np_all_time=abs(xf_cp[mid:-1])
                    self.np_all_amp_time=abs(yf_cp[mid:-1])
                else:
                    self.np_all_time=np.vstack([self.np_all_time,abs(xf_cp[mid:-1]) ])
                    self.np_all_amp_time=np.vstack([self.np_all_amp_time,abs(yf_cp[mid:-1])])



	    yf_abs=abs((yf_cp)) #redundant, change of name

	    if yf_abs[0]>yf_abs[-1]:
	    	yhf=yf_abs[0]
	    	xhf=xf_cp[0]
		poped_yf_abs=np.delete(yf_abs,0,0)
		poped_xf=np.delete(xf_cp,0,0)
	    else:
	    	yhf=yf_abs[-1]
	    	xhf=xf_cp[-1]
		poped_yf_abs=np.delete(yf_abs,-1,-1)
		poped_xf=np.delete(xf_cp,-1,-1)
	    xhf_abs=abs(xhf)
	    refined_dimensional_x = np.linspace(dimensional_x[0], dimensional_x[-1],nx*10)
	    
	    #plt.plot(dimensional_x,y)

	    yhf_fit_plot=-1.0/nx*yhf*np.sin(xhf_abs*refined_dimensional_x)
 

	    ylf = poped_yf_abs.max()
	    xlf_abs=abs(poped_xf[poped_yf_abs.argmax()])

	    ylf_fit_plot=-1.0/nx*ylf*np.sin(xlf_abs*refined_dimensional_x)

            fig_potential_fft=self.oplot_1d(var=1.0/nx*np.abs(yf_cp),xaxis=xf_cp,xlabel='freq',ylabel='amplitude',stemplot=1)
            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                os.chdir(targetdir)

                if not os.path.exists('potential_fft_along_z'):
                    os.mkdir('potential_fft_along_z')
                os.chdir('potential_fft_along_z')
                plt.savefig(filename.replace('.hdf5','.potential_fft_along_z.png'))
                plt.savefig(filename.replace('.hdf5','.potential_fft_along_z.eps'))
                os.chdir(basedir)
            if self.cb_3d_interactive.checkState()==0:
                plt.close(fig_potential_fft)

            fig_potential_along_z=self.oplot_1d(var=yhf_fit_plot,xaxis=refined_dimensional_x,xlabel='z',ylabel='potential',color='r')
            fig_potential_along_z=self.oplot_1d(var=ylf_fit_plot,fig=fig_potential_along_z,xaxis=refined_dimensional_x,xlabel='z',ylabel='potential',color='g')
            fig_potential_along_z=self.oplot_1d(var=y,fig=fig_potential_along_z,xaxis=dimensional_x,xlabel='z',ylabel='potential')

            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                os.chdir(targetdir)

                if not os.path.exists('potential_along_z'):
                    os.mkdir('potential_along_z')
                os.chdir('potential_along_z')
                plt.savefig(filename.replace('.hdf5','.potential_along_z.png'))
                plt.savefig(filename.replace('.hdf5','.potential_along_z.eps'))
                os.chdir(basedir)
            if self.cb_3d_interactive.checkState()==0:
                plt.close(fig_potential_along_z)


	    # copy to time series
            self.np_time=np.append(self.np_time,cogent_time)
            self.np_hf_time=np.append(self.np_hf_time,xhf_abs)
            self.np_hf_amp_time=np.append(self.np_hf_amp_time,yhf)
            self.np_lf_time=np.append(self.np_lf_time,xlf_abs)
            self.np_lf_amp_time=np.append(self.np_lf_amp_time,ylf)

            #print 'self.np_all_time'
            #print self.np_all_time


    
        return

    def plot_dfn(self,pathfilename,speciesname='',saveplots=1,showplots=0,x_pt=1,y_pt=1,z_pt=1,vpar_pt=1,mu_pt=1,targetdir=[]):
        self.print_ui('plot_dfn()')
        head=os.path.split(pathfilename)
        path=head[0]
        filename=head[1]
        #self.print_ui(path)
        #self.print_ui(filename)
        basedir=os.getcwd()
        if targetdir==[]:
            targetdir='./python_auto_plots'

        self.load_ghosts=(self.cb_ghosts.checkState()+self.cb_ghosts_internal.checkState())/2

        #print self.load_ghosts, type(self.load_ghosts)

        dataNd_dfn_comps,ghost,cogent_time=self.import_multdim_comps(filename=pathfilename,withghost=1 )
        title_var=r'$f_%s(\bar{v}_\parallel, \bar{\mu})$'%speciesname
        if self.ref_time!=-1.0:
            title_var=title_var+r'$(t=%.2f \mu s)$'%(cogent_time*self.ref_time*1.0E6)
        else:
            title_var=title_var+r'$(t/t_0=%.2f)$'%cogent_time

        num_cell_total_comps_tuple=dataNd_dfn_comps.shape
        #from read_input_deck import *  #for Vpar_max and Mu_max
        Vpar_max = self.read_input_var(str_var='phase_space_mapping.v_parallel_max')
        Mu_max = self.read_input_var(str_var='phase_space_mapping.mu_max')
        
        VPAR_SCALE, MU_SCALE = self.get_vpar_mu_scales(num_cell_total_comps_tuple,Vpar_max,Mu_max)
        VPAR_N, MU_N = self.get_vpar_mu_scales(num_cell_total_comps_tuple) #default max
        
        #velocity space
        if (self.cb_vpar_mu.checkState()):
            if self.gksystem_dimension==5:
                fig_dfn2d=self.plot_Nd(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],title=title_var)
            elif self.gksystem_dimension==4:
                fig_dfn2d=self.plot_Nd(dataNd_dfn_comps[x_pt,y_pt,:,:,:],title=title_var)
        #uncomment below for dfn2d plot
            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                    self.print_ui(targetdir+'/'+ 'is created.')
                os.chdir(targetdir)
                plt.savefig(filename.replace('.hdf5','.vpar_mu.png'))
                plt.savefig(filename.replace('.hdf5','.vpar_mu.eps'))
                os.chdir(basedir)
            if showplots==0 and not self.cb_vpar_mu_maxwell2D.checkState():
                plt.close(fig_dfn2d)

        
        if (self.cb_vpar_mu_smooth.checkState()):
        #uncomment below for interpolation plot
            if self.gksystem_dimension==5:
                fig_dfn2d_interp=self.plot_Nd(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],title=title_var,interpolation='spline36')
            elif self.gksystem_dimension==4:
                fig_dfn2d_interp=self.plot_Nd(dataNd_dfn_comps[x_pt,y_pt,:,:,:],title=title_var,interpolation='spline36')
            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                    self.print_ui(targetdir+'/'+ 'is created.')
                os.chdir(targetdir)
                plt.savefig(filename.replace('.hdf5','.vpar_mu_interp.png'))
                plt.savefig(filename.replace('.hdf5','.vpar_mu_interp.eps'))
                os.chdir(basedir)
            if showplots==0:
                plt.close(fig_dfn2d_interp)


        # For all mu maxwellian fitting
        ion_mass = self.read_input_var(str_var='kinetic_species.1.mass')
        elec_mass = self.read_input_var(str_var='kinetic_species.2.mass')
        t0_grid_func = self.read_input_var(str_var='.T0_grid_func.constant')
        if t0_grid_func == 0.0:
            t0_grid_func = self.read_input_var(str_var='.T0_grid_func.value')
        et0_grid_func = self.read_input_var(str_var='.eT0_grid_func.constant')
        if et0_grid_func == 0.0:
            et0_grid_func = self.read_input_var(str_var='.eT0_grid_func.value')
        nhat= self.read_input_nhat()
        Bhat= self.read_input_bhat()

        coef_maxwell,mhat,That = self.get_maxwellian_coef(pathfilename,ion_mass,t0_grid_func,elec_mass,et0_grid_func,nhat)
        
#        if self.gksystem_dimension==5:
#            data_2d_shape=dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:].shape
#        elif self.gksystem_dimension==4:
#            data_2d_shape=dataNd_dfn_comps[x_pt,y_pt,:,:,:].shape
#        data2d_maxwell=np.linspace(0.0,0.1,data_2d_shape[0]*data_2d_shape[1]*data_2d_shape[2]).reshape((data_2d_shape[0],data_2d_shape[1],data_2d_shape[2]))
#        n_fit=np.linspace(0.0,0.1,data_2d_shape[1])
#        t_fit=np.linspace(0.0,0.1,data_2d_shape[1])
#        vshift_fit=np.linspace(0.0,0.1,data_2d_shape[1])
#        for i in range(data_2d_shape[1]):
#            #sys.stdout.write('mu_ind='+str(i))
#            #sys.stdout.flush()
#            #fitted_f, n_fit[i], t_fit[i], vshift_fit[i] = get_maxwellian_fitting(coef_maxwell,mhat,That,Bhat,dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],VPAR_SCALE,MU_SCALE,mu_ind=i)
#            if self.gksystem_dimension==5:
#                fitted_f, n_fit[i], t_fit[i], vshift_fit[i] = self.get_maxwellian_fitting_with_fixed_max(mhat,That,Bhat,dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],VPAR_SCALE,MU_SCALE,mu_ind=i)
#            elif self.gksystem_dimension==4:
#                fitted_f, n_fit[i], t_fit[i], vshift_fit[i] = self.get_maxwellian_fitting_with_fixed_max(mhat,That,Bhat,dataNd_dfn_comps[x_pt,y_pt,:,:,:],VPAR_SCALE,MU_SCALE,mu_ind=i)
#            data2d_maxwell[:,i,0]=fitted_f
#
#        if (self.cb_deltaf_vpar_mu.checkState()):
#        #plot delta f
#            #data2dmax=max(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:].flatten())
#            data2dmax=max(data2d_maxwell.flatten())
#            if self.gksystem_dimension==5:
#                fig_deltaf=self.plot_Nd((dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:]-data2d_maxwell)/data2dmax,title="$(f_{COGENT}-f_M)/f_{M,Max}$",interpolation='spline36')
#            elif self.gksystem_dimension==4:
#                fig_deltaf=self.plot_Nd((dataNd_dfn_comps[x_pt,y_pt,:,:,:]-data2d_maxwell)/data2dmax,title="$(f_{COGENT}-f_M)/f_{M,Max}$",interpolation='spline36')
#            if saveplots>0:
#                if not os.path.exists(targetdir):
#                    os.mkdir(targetdir)
#                    self.print_ui(targetdir+'/'+ 'is created.')
#                os.chdir(targetdir)
#                plt.savefig(filename.replace('.hdf5','.vpar_maxwell_diff_interp.png'))
#                plt.savefig(filename.replace('.hdf5','.vpar_maxwell_diff_interp.eps'))
#                os.chdir(basedir)
#            if showplots==0:
#                plt.close(fig_deltaf)
#
#        if (self.cb_f_vpar_muat1.checkState() or  self.cb_deltaf_vpar_muat1.checkState() or  self.cb_vpar_mu_maxwell2D.checkState() or  self.cb_vpar_maxwell2D_mu_1.checkState() or  self.cb_vpar_maxwell2D_diff_mu_1.checkState() or self.cb_vpar_maxwell2D_diff.checkState() or self.cb_vpar_maxwell2D_diff_time.checkState() ):
#            mu_pt=1
#
#        if (self.cb_f_vpar_muat1.checkState()):
#            #plot difference
#            if self.gksystem_dimension==5:
#                fig_overlap=self.oplot_1d(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,:],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,:])),label='COGENT'%MU_N[mu_pt],legend=1 )
#            elif self.gksystem_dimension==4:
#                fig_overlap=self.oplot_1d(dataNd_dfn_comps[x_pt,y_pt,:,mu_pt,:],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,:,mu_pt,:])),label='COGENT'%MU_N[mu_pt],legend=1 )
#            legend_maxwellian = 'Maxw.'+r'$(n, T, V_s)$'+' = (%.2f, %.2f, %.1g)'%(n_fit[mu_pt],t_fit[mu_pt],vshift_fit[mu_pt])
#            self.oplot_1d(data2d_maxwell[:,mu_pt,0],fig_overlap,title='',linewidth=1.5, linestyle='--',color='k',label=legend_maxwellian,legend=1,ylabel='f_M, f_S')
#            if saveplots>0:
#                if not os.path.exists(targetdir):
#                    os.mkdir(targetdir)
#                    self.print_ui(targetdir+'/'+ 'is created.')
#                os.chdir(targetdir)
#                plt.savefig(filename.replace('.hdf5','.vpar_maxwell_overlap_mu_1.png'))
#                plt.savefig(filename.replace('.hdf5','.vpar_maxwell_overlap_mu_1.eps'))
#                os.chdir(basedir)
#            if showplots==0:
#                plt.close(fig_overlap)
#    
#    
#        if (self.cb_deltaf_vpar_muat1.checkState()):
#            maxatmu1=max(data2d_maxwell[:,mu_pt,0].flatten())
#            #maxwell difference plot
#            if self.gksystem_dimension==5:
#                fig_maxwell_diff=self.oplot_1d( (dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,0]-data2d_maxwell[:,mu_pt,0])/maxatmu1  ,xaxis=np.linspace(-1,1,len(data2d_maxwell[:,mu_pt,0])),label='(mu=%g)'%MU_N[mu_pt],legend=1,ylabel='(f_S - f_M)/f_{M,max}' )
#            elif self.gksystem_dimension==4:
#                fig_maxwell_diff=self.oplot_1d( (dataNd_dfn_comps[x_pt,y_pt,:,mu_pt,0]-data2d_maxwell[:,mu_pt,0])/maxatmu1  ,xaxis=np.linspace(-1,1,len(data2d_maxwell[:,mu_pt,0])),label='(mu=%g)'%MU_N[mu_pt],legend=1,ylabel='(f_S - f_M)/f_{M,max}' )
#            if saveplots>0:
#                if not os.path.exists(targetdir):
#                    os.mkdir(targetdir)
#                    self.print_ui(targetdir+'/'+ 'is created.')
#                os.chdir(targetdir)
#                plt.savefig(filename.replace('.hdf5','.vpar_maxwell_diff_at_1.png'))
#                plt.savefig(filename.replace('.hdf5','.vpar_maxwell_diff_at_1.eps'))
#                os.chdir(basedir)
#            if showplots==0:
#                plt.close(fig_maxwell_diff)


        #2dfitting
        if (self.cb_vpar_mu_maxwell2D.checkState() or  self.cb_vpar_maxwell2D_mu_1.checkState() or  self.cb_vpar_maxwell2D_diff_mu_1.checkState() or self.cb_vpar_maxwell2D_diff.checkState() or self.cb_vpar_maxwell2D_diff_time.checkState()):
            if self.gksystem_dimension==5:
                data_fitted,popt=self.get_maxwellian_fitting_2D(mhat,That,Bhat,dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,0],VPAR_SCALE,MU_SCALE)
            elif self.gksystem_dimension==4:
                data_fitted,popt=self.get_maxwellian_fitting_2D(mhat,That,Bhat,dataNd_dfn_comps[x_pt,y_pt,:,:,0],VPAR_SCALE,MU_SCALE)
            self.print_ui("Popt="+str(popt))

        if (self.cb_vpar_mu_maxwell2D.checkState()):

            if (self.cb_vpar_mu.checkState()):
                fig_maxwellian=self.oplot_2d(data_fitted[:,:,0],fig=fig_dfn2d)
            else:
                fig_maxwellain=self.oplot_2d(data_fitted[:,:,0])
            #oplot_2d(data_fitted[:,:,0],fig=fit_2d)
            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                    self.print_ui( targetdir+'/'+ 'is created.')
                os.chdir(targetdir)
                plt.savefig(filename.replace('.hdf5','.vpar_mu_maxwellian2Dfit.png'))
                plt.savefig(filename.replace('.hdf5','.vpar_mu_maxwellian2Dfit.eps'))
                os.chdir(basedir)
            if showplots==0:
                plt.close(fig_maxwellian)

        #plot difference
        #hard coding
        self.print_ui('Trying reading post processing file: finish.txt')
        wave_phase_speed = -1.0
        thermal_speed = -1.0
        fname=self.find('finish.txt', './')
        if len(fname)==0:
            self.print_ui('finish.txt file was not found!')
        else:
            with open(fname[0], 'r') as f:
                for line in f:
                    if line.lstrip().startswith('**'): #skip comment
                        continue
                    line =line.rstrip() #skip blank line
                    if not line:
                        continue 
                    else: #noncomment line
                        strippedline=line
                        if 'omega/kpar' in strippedline:
                            lhsrhs = strippedline.split("=")
                            #print lhsrhs[0],'=',lhsrhs[1],type(lhsrhs[1])
                            wave_phase_speed = float(lhsrhs[1])
                            self.print_ui(lhsrhs[0]+'='+ str(wave_phase_speed)+ str(type(wave_phase_speed)))
                        elif 'THERMAL SPEED' in strippedline:
                            lhsrhs = strippedline.split("=")
                            thermal_speed = float(lhsrhs[1])
                            self.print_ui(lhsrhs[0]+'='+ str(thermal_speed)+ str(type(thermal_speed)))
                        else:
                            continue
     

        if (self.cb_vpar_maxwell2D_mu_1.checkState()):
            if self.gksystem_dimension==5:
                fig_vpar_maxwell_2d_mu_1=self.oplot_1d(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,:],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,:])),label='COGENT'%MU_N[mu_pt],legend=1 )
            elif self.gksystem_dimension==4:
                fig_vpar_maxwell_2d_mu_1=self.oplot_1d(dataNd_dfn_comps[x_pt,y_pt,:,mu_pt,:],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,:,mu_pt,:])),label='COGENT'%MU_N[mu_pt],legend=1 )

            if wave_phase_speed != -1.0:
                normalized_wave_phase_speed = wave_phase_speed/(thermal_speed*100*Vpar_max/np.sqrt(mhat))
                plt.axvline(normalized_wave_phase_speed,color='k', linestyle=':', linewidth=1.5,label=r'$\bar{v}_\parallel=\bar{v}_{ph}=%.2f$'%(normalized_wave_phase_speed))

            legend_maxwellian = 'Maxw.'+r'$(n, T, V_s)$'+' = (%.2f, %.2f, %.1g)'%(popt[0],popt[1],popt[2])

            title_var=r'$(\bar{\mu}=%.g,$'%MU_N[mu_pt]
     ###     read computational unit time
            if self.ref_time!=-1.0:
                title_var=title_var+r'$t=%.2f \mu {s})$'%(cogent_time*self.ref_time*1.0E6)
            else:
                title_var=title_var+r'$t/t_0=%.2f)$'%cogent_time

            self.oplot_1d(data_fitted[:,mu_pt,0],fig_vpar_maxwell_2d_mu_1,title=title_var,linewidth=1.5, linestyle='--',color='k',label=legend_maxwellian,legend=1,ylabel=r'$f_%s$'%speciesname)
                  
            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                    self.print_ui( targetdir+'/'+ 'is created.')
                os.chdir(targetdir)
                plt.savefig(filename.replace('.hdf5','.vpar_maxwell2d_mu_1.png'))
                plt.savefig(filename.replace('.hdf5','.vpar_maxwell2d_mu_1.eps'))
                os.chdir(basedir)
            if showplots==0:
                plt.close(fig_vpar_maxwell_2d_mu_1)


        if (self.cb_vpar_maxwell2D_diff_mu_1.checkState() or self.cb_vpar_maxwell2D_diff_time.checkState()):
            #maxwell difference plot
            if wave_phase_speed != -1.0:
                fig=plt.figure()
                normalized_wave_phase_speed = wave_phase_speed/(thermal_speed*100*Vpar_max/np.sqrt(mhat))
                plt.axvline(normalized_wave_phase_speed,color='k', linestyle=':', linewidth=1.5,label=r'$\bar{v}_\parallel=\bar{v}_{ph}=%.2f$'%(normalized_wave_phase_speed))
            else:
                fig=plt.figure()

            title_var=r'$(\bar{\mu}=%.g,$'%MU_N[mu_pt]
            ###read computational unit time
            if self.ref_time!=-1.0:
                title_var=title_var+r'$t=%.2f \mu {s})$'%(cogent_time*self.ref_time*1.0E6)
            else:
                title_var=title_var+r'$t/t_0=%.2f)$'%cogent_time

            data_fitted_max=max(data_fitted[:,mu_pt,0].flatten())

            if self.gksystem_dimension==5:
                fig=self.oplot_1d( (dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,0]-data_fitted[:,mu_pt,0])/data_fitted_max,fig=fig,xaxis=np.linspace(-1,1,len(data_fitted[:,mu_pt,0])),label='$(f_%s-f_M)/f_{M,max}$'%speciesname,legend=1,ylabel='$(f_'+speciesname+'-f_M)/f_{M,max}$',symmetric_ylim=1, title=title_var)
            elif self.gksystem_dimension==4:
                fig=self.oplot_1d( (dataNd_dfn_comps[x_pt,y_pt,:,mu_pt,0]-data_fitted[:,mu_pt,0])/data_fitted_max,fig=fig,xaxis=np.linspace(-1,1,len(data_fitted[:,mu_pt,0])),label='$(f_%s-f_M)/f_{M,max}$'%speciesname,legend=1,ylabel='$(f_'+speciesname+'-f_M)/f_{M,max}$',symmetric_ylim=1, title=title_var)

            if self.cb_vpar_maxwell2D_diff_time.checkState():
                if len(self.np_vpar_maxwell2D_diff_time)==0:
                    self.np_vpar_maxwell2D_diff_time=(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,0]-data_fitted[:,mu_pt,0])/data_fitted_max
                    self.start_time = cogent_time
                else:
                    self.np_vpar_maxwell2D_diff_time=np.vstack([self.np_vpar_maxwell2D_diff_time,(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,0]-data_fitted[:,mu_pt,0])/data_fitted_max])
                    self.end_time = cogent_time


            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                os.chdir(targetdir)
                plt.savefig(filename.replace('.hdf5','.vpar_maxwell2d_diff_at_1.png'))
                plt.savefig(filename.replace('.hdf5','.vpar_maxwell2d_diff_at_1.eps'))
                os.chdir(basedir)
            if showplots==0:
                plt.close(fig)


        if (self.cb_vpar_maxwell2D_diff.checkState()):
        ### read computational unit time
            data_fitted_max=max(data_fitted.flatten())
            title_var='$(f_%s-f_M)/f_{M,max}$'%speciesname
            if self.ref_time!=-1.0:
                title_var=title_var+r'$(t=%.2f \mu {s})$'%(cogent_time*self.ref_time*1.0E6)
            else:
                title_var=title_var+r'$(t/t_0=%.2f)$'%cogent_time

            if self.gksystem_dimension==5:
                fig_maxwell2d_diff=self.plot_Nd( (dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:]-data_fitted)/data_fitted_max,title=title_var,interpolation='spline36',symmetric_cbar=1)
            elif self.gksystem_dimension==4:
                fig_maxwell2d_diff=self.plot_Nd( (dataNd_dfn_comps[x_pt,y_pt,:,:,:]-data_fitted)/data_fitted_max,title=title_var,interpolation='spline36',symmetric_cbar=1)
            if saveplots>0:
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                os.chdir(targetdir)
                plt.savefig(filename.replace('.hdf5','.vpar_mu_maxwell2d_diff_interp.png'))
                plt.savefig(filename.replace('.hdf5','.vpar_mu_maxwell2d_diff_interp.eps'))
                os.chdir(basedir)
            if showplots==0:
                plt.close(fig_maxwell2d_diff)

        

        if (self.cb_summation.checkState()):
            #get density by summation over velocity
            f_vpar_mu_sum = self.get_summation_over_velocity(dataNd_dfn_comps,Vpar_max,Mu_max)
            title_var='<f_%s>'%speciesname
             ### read computational unit time
            if self.ref_time!=-1.0:
                title_var=title_var+'\n(t=%.2fE-6 s)'%(cogent_time*self.ref_time*1.0E6)
            else:
                title_var=title_var+'\n(t/t0=%.2f)'%cogent_time

            fig=self.plot_Nd(f_vpar_mu_sum,title=title_var)
            #if saveplots>0:
                #if not os.path.exists(targetdir):
                #    os.mkdir(targetdir)
                #    print targetdir+'/', 'is created.'
                #else:
                #    print targetdir+'/', 'already exists.'
                #os.chdir(targetdir)
                #mlab.savefig(filename.replace('.hdf5','.f_sum_mlab_iso.png'))
                #fig.scene.save_ps(filename.replace('.hdf5','.f_sum_mlab_iso.eps'))
                #time.sleep(1)
                #os.chdir(basedir)
            #fig.scene.save_ps('fig1_mlab_iso.pdf')
            #mlab.show()
            #arr=mlab.screenshot()
            #plt.imshow(arr)
            #plt.axis('off')
            #plt.savefig('fig1_mlab_iso.eps')
            #plt.savefig('fig1_mlab_iso.pdf')
            #fig.scene.close()
            #if showplots==0:
            #    mlab.close(all=True)

            #sliced plot
            #fig=self.plot_Nd(f_vpar_mu_sum,title=title_var,x_slice=x_pt,y_slice=y_pt,z_slice=z_pt)
            #if saveplots>0:
            #    if not os.path.exists(targetdir):
            #        os.mkdir(targetdir)
            #        print targetdir+'/', 'is created.'
            #    else:
            #        print targetdir+'/', 'already exists.'
            #    os.chdir(targetdir)
            #    mlab.savefig(filename.replace('.hdf5','.f_sum_mlab_slice.png'))
            #    fig.scene.save_ps(filename.replace('.hdf5','.f_sum_mlab_slice.eps'))
            #    time.sleep(1)
            #    os.chdir(basedir)
            #if showplots==0:
            #    mlab.close(all=True)

        return
    
    def find(self, pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
           for name in files:
               if fnmatch.fnmatch(name, pattern):
                   result.append(os.path.join(root, name))
        return result

    def read_input_var(self,inputfile='',str_var='kinetic_species.2.mass'):
        if (inputfile==''):
            fname=self.find('*.in', './')
            inputfile=fname[0]
        with open(inputfile,'r') as f:
            for line in f:
                if line.lstrip().startswith('#'): #skip comment
                    continue
                line =line.rstrip() #skip blank line
                if not line:
                    continue 
                else: #noncomment line
                    strippedline=line
                    lhsrhs = strippedline.split("=")
                    l=0
                    while l<len(lhsrhs): #strip white spaces in lhs
                        lhsrhs[l]=lhsrhs[l].rstrip()
                        lhsrhs[l]=lhsrhs[l].lstrip()
                        l=l+1

                    if str_var in lhsrhs[0]:
                        result = float(lhsrhs[1])
                        return result
        return 0 #read error

    def read_input_str(self,inputfile='',str_var='.N0_grid_func.function'):
        if (inputfile==''):
            fname=self.find('*.in', './')
            inputfile=fname[0]
        with open(inputfile,'r') as f:
            for line in f:
                if line.lstrip().startswith('#'): #skip comment
                    continue
                line =line.rstrip() #skip blank line
                if not line:
                    continue 
                else: #noncomment line
                    strippedline=line
                    lhsrhs = strippedline.split("=")
                    l=0
                    while l<len(lhsrhs): #strip white spaces in lhs
                        lhsrhs[l]=lhsrhs[l].rstrip()
                        lhsrhs[l]=lhsrhs[l].lstrip()
                        l=l+1

                    if str_var in lhsrhs[0]:
                        n0_grid_func=lhsrhs[1][1:-1] #remove double quotes
                        n0_grid_func=n0_grid_func.lstrip()
                        n0_grid_func=n0_grid_func.rstrip()
                        n0_grid_func=n0_grid_func.replace('^','**')
 
                        return n0_grid_func
        return 0 #read error

    def read_input_nhat(self,inputfile=''):
        if (inputfile==''):
            fname=self.find('*.in', './')
            inputfile=fname[0]
        with open(inputfile,'r') as f:
            n0_grid_func='1.0'
            for line in f:
                if line.lstrip().startswith('#'): #skip comment
                    continue
                line =line.rstrip() #skip blank line
                if not line:
                    continue 
                else: #noncomment line
                    strippedline=line
                    lhsrhs = strippedline.split("=")
                    l=0
                    while l<len(lhsrhs): #strip white spaces in lhs
                        lhsrhs[l]=lhsrhs[l].rstrip()
                        lhsrhs[l]=lhsrhs[l].lstrip()
                        l=l+1
                    if '.N0_grid_func.function' in lhsrhs[0]:
                        #print lhsrhs[0],'=',lhsrhs[1]
                        n0_grid_func=lhsrhs[1][1:-1] #remove double quotes
                        n0_grid_func=n0_grid_func.lstrip()
                        n0_grid_func=n0_grid_func.rstrip()
                        n0_grid_func=n0_grid_func.replace('^','**')
                        self.print_ui('IN:n0_grid_func = '+n0_grid_func)

            n0_grid_func=n0_grid_func.replace('rand','1*')
            n0_grid_func=n0_grid_func.replace('atwotan','*0*')
            n0_grid_func=n0_grid_func.replace('arctan','atan')
            transformations = (standard_transformations + (implicit_multiplication_application,))
            pe=parse_expr(n0_grid_func,transformations=transformations)
            f_n0_grid_func= lambdify((x,y,z),pe)
            in_f_n0 = f_n0_grid_func(0,0,0)
            if in_f_n0 >0.0:
                nhat = in_f_n0
                self.print_ui('nhat = '+ str(nhat))
            else:
                nhat =1.0
                self.print_ui('nhat(default) = '+ str(nhat))
            return nhat
        return 0 #read error

    def read_input_bhat(self,inputfile=''):
        if (inputfile==''):
            fname=self.find('*.in', './')
            inputfile=fname[0]
        with open(inputfile,'r') as f:
            bz_found=0
            by_found=0
            btor_scale_found=0
            for line in f:
                if line.lstrip().startswith('#'): #skip comment
                    continue
                line =line.rstrip() #skip blank line
                if not line:
                    continue 
                else: #noncomment line
                    strippedline=line
                    lhsrhs = strippedline.split("=")
                    l=0
                    while l<len(lhsrhs): #strip white spaces in lhs
                        lhsrhs[l]=lhsrhs[l].rstrip()
                        lhsrhs[l]=lhsrhs[l].lstrip()
                        l=l+1

                    if 'gksystem.magnetic_geometry_mapping.slab.Bz_inner' in lhsrhs[0]:
                        bz_inner = float(lhsrhs[1])
                        bz_found = 1
                        self.print_ui('IN:bz_inner = '+str(bz_inner))
                    if 'gksystem.magnetic_geometry_mapping.slab.By_inner' in lhsrhs[0]:
                        by_inner = float(lhsrhs[1])
                        by_found = 1
                        self.print_ui('IN:by_inner = '+str(by_inner))
                    if 'gksystem.magnetic_geometry_mapping.miller.Btor_scale' in lhsrhs[0]:
                        btor_scale= float(lhsrhs[1])
                        btor_scale= 1

 
            return np.sqrt(bz_inner**2+by_inner**2)
        return 0 #read error

    def get_vpar_mu_scales(self,num_cell_total_comps_tuple,arg_Vpar_max=1.0,arg_Mu_max=1.0):
        #num_cell_total_comps_tuple[-3] = vpar cell
        #num_cell_total_comps_tuple[-2] = mu cell
        vpar_cell_dim_begin = -1.0*arg_Vpar_max+(arg_Vpar_max*1.0+arg_Vpar_max*1.0)/float(num_cell_total_comps_tuple[-3])/2.0
        vpar_cell_dim_end   =  1.0*arg_Vpar_max-(arg_Vpar_max*1.0+arg_Vpar_max*1.0)/float(num_cell_total_comps_tuple[-3])/2.0
        mu_cell_dim_begin = arg_Mu_max*0.0+(arg_Mu_max*1.0-arg_Mu_max*0.0)/float(num_cell_total_comps_tuple[-2])/2.0
        mu_cell_dim_end   = arg_Mu_max*1.0-(arg_Mu_max*1.0-arg_Mu_max*0.0)/float(num_cell_total_comps_tuple[-2])/2.0
        VPAR_CELL,MU_CELL = np.mgrid[vpar_cell_dim_begin:vpar_cell_dim_end:(num_cell_total_comps_tuple[-3]*1j),mu_cell_dim_begin:mu_cell_dim_end:(num_cell_total_comps_tuple[-2]*1j)]
        
        #VPAR_SCALE = VPAR_CELL[:,0]*np.sqrt(mhat) #for trunk
        VPAR_SCALE = VPAR_CELL[:,0] #for mass dependent normalization
        MU_SCALE = MU_CELL[0,:]
        return VPAR_SCALE, MU_SCALE
    
    def plot_Nd(self,var,ghostIn=[],title='',xlabel='xlabel',ylabel='ylabel',xaxis=[],wh=1,fig_size_x=800,fig_size_y=600,sliced=0,x_slice=-1,y_slice=-1,z_slice=-1,interpolation='none',label='',symmetric_cbar=0):
        self.print_ui('plot_Nd()')
        #wh=1 # 0: black background, 1: whithe background
        #first check the rank of input data
        var_shape=var.shape
        var_components=var_shape[-1]
        var_dim=len(var_shape)-1

        #set environment
        if ghostIn==[]: #no ghost cell
            #no ghost input
            #axes numbering
            range_var=[0]*var_dim*2
            for i in range(var_dim):
                range_var[2*i]=0
                range_var[2*i+1]=1
            #axes boundary
            bounds_var_ax = [1]*(2*(len(var.shape)-1))
            for i in range((len(var.shape)-1)):
                bounds_var_ax[2*i]=1
                bounds_var_ax[2*i+1]=var.shape[i]
            #outline boundary
            bounds_var_ol = [1]*(2*(len(var.shape)-1))
            #print 'bounds_var_ol=',bounds_var_ol
            for i in range((len(var.shape)-1)):
                bounds_var_ol[2*i]=1
                bounds_var_ol[2*i+1]=var.shape[i]
            ax_line_width=1.0
            ol_line_width=1.0
     
        else: #ghost cell
            dghost=np.zeros(len(ghostIn))
            for i in range(len(dghost)):
                dghost[i]=float(ghostIn[i])/(var.shape[i]-2*ghostIn[i])
            #axes numbering
            range_var=[0]*var_dim*2
            for i in range(var_dim):
                range_var[2*i]=0-dghost[i]
                range_var[2*i+1]=1+dghost[i]
            #axes boundary
            bounds_var_ax = [1]*(2*(len(var.shape)-1))
            for i in range((len(var.shape)-1)):
                bounds_var_ax[2*i]=1
                bounds_var_ax[2*i+1]=var.shape[i]
            #outline boundary
            bounds_var_ol = [1]*(2*(len(var.shape)-1))
            for i in range((len(var.shape)-1)):
                bounds_var_ol[2*i]=1+ghostIn[i]
                bounds_var_ol[2*i+1]=var.shape[i]-ghostIn[i]
            ax_line_width=1.0
            ol_line_width=2.0

        if x_slice==-1:#default slice middle
            x_slice_pt=float(bounds_var_ol[2*0+1])/2+float(bounds_var_ol[2*0])/2-1
        else:
            if type(x_slice)==type(1):#consider it as point
                self.print_ui( 'x_slice='+ str(x_slice)+ str(type(x_slice)))
                x_slice_pt=float(bounds_var_ol[2*0])+x_slice-1
            else: #considier it a number between 0.0 to 1.0
                x_slice_pt=float(bounds_var_ol[2*0])+(float(bounds_var_ol[2*0+1])-float(bounds_var_ol[2*0]))*x_slice-1
        if var_dim>1:
            if y_slice==-1:#default slice middle
                y_slice_pt=float(bounds_var_ol[2*1+1])/2+float(bounds_var_ol[2*1])/2-1
            else:
                if type(y_slice)==type(1):#consider it as point
                    self.print_ui('y_slice='+ str(y_slice)+ str(type(y_slice)))
                    y_slice_pt=float(bounds_var_ol[2*1])+y_slice-1
                else: #considier it a number between 0.0 to 1.0
                    y_slice_pt=float(bounds_var_ol[2*1])+(float(bounds_var_ol[2*1+1])-float(bounds_var_ol[2*1]))*y_slice-1
        if var_dim>2:
            if z_slice==-1:#default slice middle
                z_slice_pt=float(bounds_var_ol[2*2+1])/2+float(bounds_var_ol[2*2])/2-1
            else:
                if type(z_slice)==type(1):#consider it as point
                    self.print_ui('z_slice='+str(z_slice)+ str(type(z_slice)))
                    z_slice_pt=float(bounds_var_ol[2*2])+z_slice-1
                else: #considier it a number between 0.0 to 1.0
                    z_slice_pt=float(bounds_var_ol[2*2])+(float(bounds_var_ol[2*2+1])-float(bounds_var_ol[2*2]))*z_slice-1

        #print bounds_var_ax
        #print bounds_var_ol
        #print x_slice_pt,x_slice
        #if var_dim>1:
        #    print y_slice_pt,y_slice
        #if var_dim>2:
        #    print z_slice_pt,z_slice

     
        #Start plotting   
        if var_dim==3:
            self.print_ui('var_dim = '+str(var_dim))
            s3 = slice3(var[:,:,:,0],self,interactive=self.cb_3d_interactive.checkState())
            s3.xlabel('x',fontsize=18)
            s3.ylabel('y',fontsize=18)
            s3.zlabel('z',fontsize=18)
            if self.cb_3d_interactive.checkState():
                s3.show()

            return s3.fig


            #try 3D plot using mayavi
            #if var_components>1:
                #try vector field plot (x,y,z) = 0 2 -1
                #fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
                #src=mlab.pipeline.vector_field(var[:,:,:,0],var[:,:,:,2],-var[:,:,:,1])
                
                #vh=mlab.pipeline.vectors(src,mask_points=4)
                #engine=mlab.get_engine()
                #vectors = engine.scenes[0].children[0].children[0].children[0]
                #vectors.glyph.glyph_source.glyph_source = vectors.glyph.glyph_source.glyph_dict['arrow_source']
                #vh.glyph.glyph_source.glyph_source = vh.glyph.glyph_source.glyph_dict['arrow_source']
                #vh.glyph.glyph.scale_factor = 1.0

                #engine=mlab.get_engine()
                #s=engine.current_scene
                #module_manager = s.children[0].children[0]
                #module_manager.vector_lut_manager.show_scalar_bar = True
                #module_manager.vector_lut_manager.show_legend = True
                #module_manager.vector_lut_manager.scalar_bar.title = title
                #module_manager.vector_lut_manager.scalar_bar_representation.position2 = np.array([ 0.1,  0.8])
                #module_manager.vector_lut_manager.scalar_bar_representation.position = np.array([ 0.05,  0.1])
                #module_manager.vector_lut_manager.label_text_property.color = (1-wh,1-wh, 1-wh)
                #module_manager.vector_lut_manager.title_text_property.color = (1-wh, 1-wh, 1-wh)


            #elif sliced==0 and (x_slice==-1 and y_slice==-1 and z_slice==-1):
                #try iso plot 
                #fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
                #ch=mlab.contour3d(var[:,:,:,0],contours=10,transparent=True,opacity=0.8)
                #cb=mlab.colorbar(title=title,orientation='vertical' )
                #cb.title_text_property.color=(1-wh,1-wh,1-wh)
                #cb.label_text_property.color=(1-wh,1-wh,1-wh)

            #else:
                ##try slice plot 
                #fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
                #if sliced>0:
                #    sxh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='x_axes',slice_index=x_slice_pt)
                #    sxh.ipw.slice_position=x_slice_pt+1
                #    syh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='y_axes',slice_index=y_slice_pt)
                #    syh.ipw.slice_position=y_slice_pt+1
                #    szh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='z_axes',slice_index=z_slice_pt)
                #    szh.ipw.slice_position=z_slice_pt+1
                #else:
                #    if x_slice>-1:
                #        sxh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='x_axes',slice_index=x_slice_pt)
                #        sxh.ipw.slice_position=x_slice_pt+1
                #    if y_slice>-1:
                #        syh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='y_axes',slice_index=y_slice_pt)
                #        syh.ipw.slice_position=y_slice_pt+1
                #    if z_slice>-1:
                #        szh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='z_axes',slice_index=z_slice_pt)
                #        szh.ipw.slice_position=z_slice_pt+1

                #cb=mlab.colorbar(title=title,orientation='vertical' )
                #cb.title_text_property.color=(1-wh,1-wh,1-wh)
                #cb.label_text_property.color=(1-wh,1-wh,1-wh)


            #ax = mlab.axes(nb_labels=5,ranges=range_var)
            #ax.axes.property.color=(1-wh,1-wh,1-wh)
            #ax.axes.property.line_width = ax_line_width
            #ax.axes.bounds=(bounds_var_ax)
            #ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
            #ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
            #ax.axes.label_format='%.2f'
            #ol=mlab.outline(color=(1-wh,1-wh,1-wh),extent=bounds_var_ol)
            #ol.actor.property.line_width = ol_line_width
            #mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')
            #return fig

        elif var_dim==5:
            if var_components>1:
                self.print_ui('dfn plot with mult components')
                return
            else:
                self.print_ui('dfn plot with single component')
                #try collect for density

                return 
        elif var_dim==2:
            #possibly vpar mu plot, only plot first component
            self.init_plotting()
            fig=plt.figure()
            plt.subplot(111)
            #plt.gca().margins(0.1, 0.1)
            if symmetric_cbar>0:
                v_min = var[:,:,0].min()
                v_max = var[:,:,0].max()
                if v_min<0 and v_max>0:
                    v_min=-max(abs(v_min),abs(v_max))
                    v_max=max(abs(v_min),abs(v_max))
                    v_max=max(abs(v_min),abs(v_max))
                    im=plt.imshow(var[:,:,0].T,vmin=v_min,vmax=v_max,interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
                else:
                    im=plt.imshow(var[:,:,0].T,interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
            else:
                im=plt.imshow(var[:,:,0].T,interpolation=interpolation,origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
            if title=='':
                pass
            else:
                plt.title(title)
            if xlabel=='xlabel':
                plt.xlabel(r'$\bar{v}_\parallel$')
            else:
                plt.xlabel(xlabel)
            if ylabel=='ylabel':
                plt.ylabel(r'$\bar{\mu}$')
            else:
                plt.ylabel(ylabel)
            self.add_colorbar(im,field=var[:,:,0])
            #plt.tight_layout()
            fig.set_tight_layout(True)
            return fig
        elif var_dim==1:
            #simple line out plot
            self.print_ui('Try using oplot_1d')
            self.init_plotting()
            fig=plt.figure()
            plt.subplot(111)
            #plt.gca().margins(0.1, 0.1)
            if xaxis==[]:
                im=plt.plot(range(len(var[:,0])),var[:,0],label=label)
            elif len(xaxis)==len(var[:,0]):
                im=plt.plot(xaxis,var[:,0],label=label)
            else:
                im=plt.plot(range(len(var[:,0])),var[:,0],label=label)
            if title=='':
                pass
            else:
                plt.title(title)
            if xlabel=='xlabel':
                plt.xlabel(r'$\bar{v}_\parallel$')
            else:
                plt.xlabel(xlabel)
            if ylabel=='ylabel':
                plt.ylabel(r'$f$')
            else:
                plt.ylabel(ylabel)
            #plt.tight_layout()
            fig.set_tight_layout(True)
            return fig


        return 

    def init_plotting(self,form=''):
        graphDPI =200
        if (form == '2x3'):
            plt.rcParams['figure.figsize'] = (16, 9)
        elif (form == '2x2'):
            plt.rcParams['figure.figsize'] = (12, 9)
        else:
            plt.rcParams['figure.figsize'] = (4, 3)
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.labelsize'] = 1.2*plt.rcParams['font.size']
            plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
            plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
            plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
            plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
            plt.rcParams['savefig.dpi'] = graphDPI
            plt.rcParams['xtick.major.size'] = 3
            plt.rcParams['xtick.minor.size'] = 3
            plt.rcParams['xtick.major.width'] = 1
            plt.rcParams['xtick.minor.width'] = 1
            plt.rcParams['ytick.major.size'] = 3
            plt.rcParams['ytick.minor.size'] = 3
            plt.rcParams['ytick.major.width'] = 1
            plt.rcParams['ytick.minor.width'] = 1
            plt.rcParams['legend.frameon'] = False
            plt.rcParams['legend.loc'] = 'center left'
            plt.rcParams['axes.linewidth'] = 1

    def add_colorbar_dual_u(self, im, aspect=20, pad_fraction=0.5, field=[], **kwargs):
        """Add a vertical color bar to an image plot."""
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0/aspect)
        print width
        print type(width)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        cb = im.axes.figure.colorbar(im, cax=cax, **kwargs)
        #change number of ticks
        m0=np.nanmin(field)            # colorbar min value
        m4=np.nanmax(field)            # colorbar max value
        num_ticks=5
        ticks = np.linspace(m0, m4, num_ticks)
        labels = np.linspace(m0, m4, num_ticks)
        labels_math=[self.latex_float(i) for i in labels]
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels_math)
        
        cb.update_ticks()
        return cb

    def add_colorbar(self, im, aspect=20, pad_fraction=0.5, field=[], **kwargs):
        """Add a vertical color bar to an image plot."""
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0/aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        cb = im.axes.figure.colorbar(im, cax=cax, **kwargs)
        #change number of ticks
        m0=field.min()            # colorbar min value
        m4=field.max()             # colorbar max value
        num_ticks=5
        ticks = np.linspace(m0, m4, num_ticks)
        labels = np.linspace(m0, m4, num_ticks)
        labels_math=[self.latex_float(i) for i in labels]
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels_math)
        
        cb.update_ticks()
        return cb

    def latex_float(self,f):
        float_str = "{0:.2g}".format(f)
        #float_str = "{0:.6g}".format(f)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
        else:
            return r"$%.2g$"%f
    
    def get_maxwellian_coef(self,dfnfilename,mi,ti,me,te,nhat):
        if 'electron' in dfnfilename:
            mhat = me
            That = te
            coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)
            self.print_ui('electron (coef, mhat, That) = '+ str(coef_maxwell)+ str(mhat)+ str(That))
            return coef_maxwell,mhat,That
        elif 'hydrogen' in dfnfilename:
            mhat = mi
            That = ti
            coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)
            self.print_ui( 'hydrogen (coef, mhat, That) = '+ str(coef_maxwell)+ str(mhat)+ str(That))
            return coef_maxwell,mhat,That
        else:
            mhat = 1.0
            That = 1.0
            coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)
            self.print_ui( 'default (coef, mhat, That) = '+ str(coef_maxwell)+ str(mhat)+ str(That))
            return coef_maxwell,mhat,That
     
#    def get_maxwellian_fitting_with_fixed_max(self,mhat,That,Bhat,data_dfn,VPAR_SCALE,MU_SCALE,mu_ind=0):
#        #least square fitting on a slice of MU index=0
#        guess_temp =1.0/(2.0*That)
#        guess_shift =0.0
#        fixed_max_den=max(data_dfn[:,mu_ind,0])
#        optimize_func = lambda z: fixed_max_den*np.exp(-z[0]*(VPAR_SCALE-z[1])**2)-data_dfn[:,mu_ind,0] 
#        
#        est_temp, est_shift = leastsq(optimize_func, [guess_temp, guess_shift])[0]
#        fitted_f = fixed_max_den*np.exp(-est_temp*(VPAR_SCALE-est_shift)**2)
#        t_fit = 1.0/(est_temp*2.0)
#        n_fit = fixed_max_den*np.sqrt(np.pi)/(0.5*mhat/t_fit)**(1.5)/np.exp(-MU_SCALE[mu_ind]*Bhat/2.0/t_fit)
#        vshift_fit = est_shift
#        #self.print_ui('(n_fit, t_fit, vshift_fit)='+str(n_fit)+str(t_fit)+str(vshift_fit))
#        return fitted_f, n_fit, t_fit, vshift_fit
    
    def get_maxwellian_fitting_2D(self,mhat,That,Bhat,data_dfn,VPAR_SCALE,MU_SCALE):

        fixed_max_den=max(data_dfn[:,0])
        n_guess = fixed_max_den*np.sqrt(np.pi)/(0.5*mhat/That)**(1.5)/np.exp(-MU_SCALE[0]*Bhat/2.0/That)
        vs_guess =0.0
        t_guess =That
        initial_guess = (n_guess,t_guess,vs_guess)

        self.print_ui("initial_guess="+str(initial_guess))
        
        self.print_ui("VPAR_SCALE.shape="+str(VPAR_SCALE.shape))
        self.print_ui("MU_SCALE.shape="+ str(MU_SCALE.shape))

        x,y = np.meshgrid(VPAR_SCALE,MU_SCALE)
        self.print_ui("x.shape="+str(x.shape))
        self.print_ui("y.shape="+str(y.shape))
        
        self.print_ui("data_dfn.shape="+str(data_dfn.shape))

        popt, pcov = curve_fit(self.twoD_Gaussian_wrapper, (x, y,Bhat,mhat), data_dfn.T.ravel(), p0=initial_guess)
        data_fitted_raveled = self.twoD_Gaussian_wrapper((x,y,Bhat,mhat),*popt)
        data_fitted = data_fitted_raveled.reshape(len(MU_SCALE),len(VPAR_SCALE)).T
        data_fitted = data_fitted.reshape(len(VPAR_SCALE),len(MU_SCALE),1)
        self.print_ui("data_fitted.shape="+str(data_fitted.shape))

        return data_fitted, popt
    
    def twoD_Gaussian_wrapper(self, (x, y,b,m), n0, t0, vs0):
        vs0 = float(vs0)
        b = float(b)
        m = float(m)
        a = 1.0/(2*t0)
        c = b/(2*t0)
        amplitude = n0/np.sqrt(np.pi)*(m/(2.0*t0))**(1.5)
        g = amplitude*np.exp( - (a*((x-vs0)**2) + c*(y)) )
        return g.ravel()
    
    def oplot_2d(self,var,x=[],y=[],fig=[],ghostIn=[],title='',xlabel='xlabel',ylabel='ylabel',wh=1,fig_size_x=800,fig_size_y=600,linewidth=1.5,linestyle='-',color='b',label='',legend=[]):
        #consider vpar-mu simple plot 
        if fig==[]:
            fig=plt.figure()
        else:
            fig=plt.figure(fig.number)
        ax1=fig.gca()
        if x==[] or y==[]:
            x=np.linspace(-1,1,var.shape[0])
            y=np.linspace( 0,1,var.shape[1])
            x,y = np.meshgrid(x,y)
        ax1.contour(x,y,var.T, 8 , label=label)
        return fig
    
    def oplot_1d(self,var,fig=[],ghostIn=[],title='',xlabel='xlabel',ylabel='ylabel',xaxis=[],wh=1,fig_size_x=800,fig_size_y=600,linewidth=1.5,linestyle='-',color='b',label='',legend=[],symmetric_ylim=0,stemplot=0):
        #consider vpar-f simple plot 
        if fig==[]:
            fig=plt.figure()
        else:
            fig=plt.figure(fig.number)
        ax1=plt.gca()
        if xaxis==[]:
            xaxis=np.linspace(-1,1,len(var))

        if stemplot==1:
            ax1.stem(xaxis,var,linewidth=linewidth,linestyle=linestyle,color=color,label=label)
        else:
            ax1.plot(xaxis,var,linewidth=linewidth,linestyle=linestyle,color=color,label=label)
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

        if title=='':
            pass 
        else:
            plt.title(title)
        if xlabel=='xlabel':
            plt.xlabel(r'$\bar{v}_\parallel$')
        else:
            plt.xlabel(xlabel)
        if ylabel=='ylabel':
            plt.ylabel(r'$f$')
        else:
            plt.ylabel(ylabel)

        if symmetric_ylim>0:
            ymin, ymax = ax1.get_ylim()
            if ymin<0 and ymax>0:
               ax1.set_ylim([-max(abs(ymin), abs(ymax)),max(abs(ymin),abs(ymax))])
               #tick format
               scale_pow =np.log10(ymax).astype(int) 
               ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x,p :"%.2f"%(x/(10**scale_pow))))
               plt.ylabel(ylabel+r'$(\times 10^{{{0:d}}})$'.format(scale_pow))

        if legend==[]:
            pass
        else:
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim([ymin, ymax*1.25])

            handles, labels = ax1.get_legend_handles_labels()
            lgd = ax1.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.0,1.0))
            #lgd = ax1.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0,1.0))
            #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

        #plt.tight_layout()
        fig.set_tight_layout(True)
        return fig
    
    def get_summation_over_velocity(self,dataNd_dfn_comps,Vpar_max,Mu_max):
        self.print_ui('get_summation_over_velocity()')
        num_cell_total_comps_tuple=dataNd_dfn_comps.shape
        #num_cell_total_comps_tuple[-6] = x cell
        #num_cell_total_comps_tuple[-5] = y cell ; x for 4D
        #num_cell_total_comps_tuple[-4] = z cell ; y for 4D
        #num_cell_total_comps_tuple[-3] = vpar cell
        #num_cell_total_comps_tuple[-2] = mu cell
        #num_cell_total_comps_tuple[-1] = comps
        dim=len(num_cell_total_comps_tuple)-1
        self.print_ui( 'dim='+str(dim))

        #density sum
        if dim==4:
            num_xcell=num_cell_total_comps_tuple[-5]
            num_ycell=num_cell_total_comps_tuple[-4]
            num_vparcell=num_cell_total_comps_tuple[-3]
            num_mucell=num_cell_total_comps_tuple[-2]
            num_compcell=num_cell_total_comps_tuple[-1]
            f_vpar_mu_sum = np.zeros((num_xcell,num_ycell,num_compcell))
            delta_Mu = 1.0*Mu_max/num_mucell
            delta_Vpar = 2.0*Vpar_max/num_vparcell
            for d in range(num_compcell):
                for i in range(num_xcell):
                    for j in range(num_ycell):
                        sumovervparandmu=0.0
                        for k in range(num_vparcell):
                            sumovermu=0.0
                            for l in range(num_mucell):
                                sumovermu=sumovermu+dataNd_dfn_comps[i,j,k,l,d]
                            sumovermu=sumovermu*delta_Mu
                            sumovervparandmu=sumovervparandmu+sumovermu
                        sumovervparandmu=sumovervparandmu*delta_Vpar 
                        f_vpar_mu_sum[i,j,d]=f_vpar_mu_sum[i,j,d]+sumovervparandmu
        elif dim==5:
            num_xcell=num_cell_total_comps_tuple[-6]
            num_ycell=num_cell_total_comps_tuple[-5]
            num_zcell=num_cell_total_comps_tuple[-4]
            num_vparcell=num_cell_total_comps_tuple[-3]
            num_mucell=num_cell_total_comps_tuple[-2]
            num_compcell=num_cell_total_comps_tuple[-1]
            f_vpar_mu_sum = np.zeros((num_xcell,num_ycell,num_zcell,num_compcell))
            delta_Mu = 1.0*Mu_max/num_mucell
            delta_Vpar = 2.0*Vpar_max/num_vparcell
            total_num = num_compcell*num_xcell
            cnt=0
            self.pb_download.setValue(0)
            for d in range(num_compcell):
                for i in range(num_xcell):
                    cnt=cnt+1
                    self.printProgress(cnt,total_num)
                    for j in range(num_ycell):
                        for k in range(num_zcell):
                            sumovervparandmu=0.0
                            for l in range(num_vparcell):
                                sumovermu=0.0
                                for m in range(num_mucell):
                                    sumovermu=sumovermu+dataNd_dfn_comps[i,j,k,l,m,d]
                                sumovermu=sumovermu*delta_Mu
                                sumovervparandmu=sumovervparandmu+sumovermu
                            sumovervparandmu=sumovervparandmu*delta_Vpar 
                            f_vpar_mu_sum[i,j,k,d]=f_vpar_mu_sum[i,j,k,d]+sumovervparandmu
        self.print_ui(str(f_vpar_mu_sum.shape))
        return f_vpar_mu_sum


class slice3(object):
    def __init__(self,var,mother,interactive=0):
        nx = len(var[:,0,0])
        ny = len(var[0,:,0])
        nz = len(var[0,0,:])

        self.x = np.linspace(0,1,nx)
        self.y = np.linspace(0,1,ny)
        self.z = np.linspace(0,1,nz)
        self.data =var

        self.fig = plt.figure(1,(20,7))
        self.ax1 = self.fig.add_subplot(131,aspect='equal')
        self.ax2 = self.fig.add_subplot(132,aspect='equal')
        self.ax3 = self.fig.add_subplot(133,aspect='equal')

        #self.xslice = self.ax1.imshow(var[0,:,:],extent=(self.z[0],self.z[-1],self.y[0],self.y[-1]))
        #self.yslice = self.ax2.imshow(var[:,0,:],extent=(self.z[0],self.z[-1],self.x[0],self.x[-1]))
        #self.zslice = self.ax3.imshow(var[:,:,0],extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]))
        self.xslice = self.ax1.imshow(np.flipud(var[int(mother.te_x_pt.toPlainText())-1,:,:]),extent=(self.z[0],self.z[-1],self.y[0],self.y[-1]))
        self.yslice = self.ax2.imshow(np.flipud(var[:,int(mother.te_y_pt.toPlainText())-1,:]),extent=(self.z[0],self.z[-1],self.x[0],self.x[-1]))
        self.zslice = self.ax3.imshow(np.flipud((var[:,:,int(mother.te_z_pt.toPlainText())-1]).transpose()),extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]))



        self.xplot_zline = self.ax1.axvline(color='m',linestyle='--',lw=2)
        #self.xplot_zline.set_xdata(self.z[0]) 
        self.xplot_zline.set_xdata(int(mother.te_z_pt.toPlainText())-1) 

        self.xplot_yline = self.ax1.axhline(color='m',linestyle='--',lw=2)
        #self.xplot_yline.set_ydata(self.y[0])
        self.xplot_yline.set_ydata(int(mother.te_y_pt.toPlainText())-1)

        self.yplot_xline = self.ax2.axhline(color='m',linestyle='--',lw=2)
        #self.yplot_xline.set_ydata(self.x[0])
        self.yplot_xline.set_ydata(int(mother.te_x_pt.toPlainText())-1)

        self.yplot_zline = self.ax2.axvline(color='m',linestyle='--',lw=2)
        #self.yplot_zline.set_xdata(self.z[0])
        self.yplot_zline.set_xdata(int(mother.te_z_pt.toPlainText())-1)

        self.zplot_xline = self.ax3.axvline(color='m',linestyle='--',lw=2)
        #self.zplot_xline.set_xdata(self.x[0])
        self.zplot_xline.set_xdata(int(mother.te_x_pt.toPlainText())-1)

        self.zplot_yline = self.ax3.axhline(color='m',linestyle='--',lw=2)
        #self.zplot_yline.set_ydata(self.y[0])
        self.zplot_yline.set_ydata(int(mother.te_y_pt.toPlainText())-1)


        # draw
        self.update_x(int(mother.te_x_pt.toPlainText())-1)
        self.update_y(int(mother.te_y_pt.toPlainText())-1)
        self.update_z(int(mother.te_z_pt.toPlainText())-1)

      
        if interactive!=0:
            # Create and initialize x-slider
            self.sliderax1 = self.fig.add_axes([0.125,0.06,0.225,0.04])
            #self.sliderax1 = self.fig.add_axes([0.125,0.08,0.225,0.03])
            #self.sliderx = DiscreteSlider(self.sliderax1,'',0,len(self.x)-1,increment=1,valinit=0)
            #self.sliderx = PageSlider(self.sliderax1,'x',numpages=len(self.x),activecolor="orange")
            self.sliderx = PageSlider(self.sliderax1,'x',numpages=len(self.x),val_cb=int(mother.te_x_pt.toPlainText())-1,activecolor="orange")
            #self.sliderx = PageSlider(self.sliderax1,'x',numpages=len(self.x),valinit=int(mother.te_x_pt.toPlainText()),activecolor="orange")
            self.sliderx.on_changed(self.update_x)
            self.sliderx.set_val(int(mother.te_x_pt.toPlainText())-1)
            # Create and initialize y-slider
            self.sliderax2 = self.fig.add_axes([0.4,0.06,0.225,0.04])
            #self.sliderax2 = self.fig.add_axes([0.4,0.08,0.225,0.03])
            #self.slidery = DiscreteSlider(self.sliderax2,'',0,len(self.y)-1,increment=1,valinit=0)
            #self.slidery = PageSlider(self.sliderax2,'y',numpages=len(self.y),activecolor="orange")
            self.slidery = PageSlider(self.sliderax2,'y',numpages=len(self.y),val_cb=int(mother.te_y_pt.toPlainText())-1,activecolor="orange")
            self.slidery.on_changed(self.update_y)
            #self.slidery.set_val(0)
            self.slidery.set_val(int(mother.te_y_pt.toPlainText())-1)
            # Create and initialize z-slider
            self.sliderax3 = self.fig.add_axes([0.675,0.06,0.225,0.04])
            #self.sliderax3 = self.fig.add_axes([0.675,0.08,0.225,0.03])
            #self.sliderz = DiscreteSlider(self.sliderax3,'',0,len(self.z)-1,increment=1,valinit=0)
            #self.sliderz = PageSlider(self.sliderax3,'z',numpages=len(self.z),activecolor="orange")
            self.sliderz = PageSlider(self.sliderax3,'z',numpages=len(self.z),val_cb=int(mother.te_z_pt.toPlainText())-1,activecolor="orange")
            self.sliderz.on_changed(self.update_z)
            #self.sliderz.set_val(0)
            self.sliderz.set_val(int(mother.te_z_pt.toPlainText())-1)



        # Make plots square
        z0,z1 = self.ax1.get_xlim()
        x0,x1 = self.ax2.get_ylim()
        y0,y1 = self.ax1.get_ylim()
        self.ax1.set_aspect((z1-z0)/(y1-y0))
        self.ax2.set_aspect((z1-z0)/(x1-x0))
        self.ax3.set_aspect((x1-x0)/(y1-y0))    


    def xlabel(self,*args,**kwargs):
        self.ax2.set_ylabel(*args,**kwargs)
        self.ax3.set_xlabel(*args,**kwargs)

    def ylabel(self,*args,**kwargs):
        self.ax1.set_ylabel(*args,**kwargs)
        self.ax3.set_ylabel(*args,**kwargs)
  
    def zlabel(self,*args,**kwargs):
        self.ax1.set_xlabel(*args,**kwargs)
        self.ax2.set_xlabel(*args,**kwargs) 

    def update_x(self,value): 
        self.xslice.set_data(np.flipud(self.data[int(value),:,:]))  
        self.yplot_xline.set_ydata(self.x[int(value)])
        self.zplot_xline.set_xdata(self.x[int(value)])

    def update_y(self,value): 
        self.yslice.set_data(np.flipud(self.data[:,int(value),:]))  
        self.xplot_yline.set_ydata(self.y[int(value)])
        self.zplot_yline.set_ydata(self.y[int(value)])

    def update_z(self,value): 
        self.zslice.set_data(np.flipud((self.data[:,:,int(value)]).transpose()))
        self.xplot_zline.set_xdata(self.z[int(value)])
        self.yplot_zline.set_xdata(self.z[int(value)])


    def show(self):
        self.fig.show()


class  DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps and no text. 
       Modified from a class created by Joe Kington and submitted to StackOverflow on Dec 1 2012
       http://stackoverflow.com/questions/13656387/can-i-make-matplotlib-sliders-more-discrete
    """
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized to."""
        self.inc = kwargs.pop('increment', 1)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):

        xy = self.poly.xy
        xy[2] = val, 1
        xy[3] = val, 0
        self.poly.xy = xy
  
        # Suppress slider label
        self.valtext.set_text('')

        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.iteritems():
            func(val)

class PageSlider(matplotlib.widgets.Slider):

    def __init__(self, ax, label, numpages = 10, valinit=0, valfmt='%1d', 
                 closedmin=True, closedmax=True,  
                 dragging=True, val_cb=1, **kwargs):

        self.facecolor=kwargs.get('facecolor',"w")
        self.activecolor = kwargs.pop('activecolor',"b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = numpages
        self.val_cb = val_cb

        super(PageSlider, self).__init__(ax, label, 0, numpages, 
                            valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = matplotlib.patches.Rectangle((float(i)/numpages, 0), 1./numpages, 1, 
                                transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i)/numpages+0.5/numpages, 0.5, str(i+1),  
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax, label=ur'$\u25C0$', 
                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax, label=ur'$\u25B6$', 
                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

        self.update()

    def update(self):
        i = int(self.val_cb)
        if i >=self.valmax:
            return
        self._colorize(i)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i+1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i-1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)




if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


