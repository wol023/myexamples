import sys
import os
from PyQt4 import QtCore, QtGui, uic


qtCreatorFile = "plot_calc.ui" # Enter file here.
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.open_target_button.clicked.connect(self.print_target_file_lines)
        self.button_plot_selected.clicked.connect(self.plotFigure)

    def check_and_try_cluster(self,selected_file,status=0):
        ###check cluster
        head=os.path.split((selected_file))
        path_loc=head[0]
        file_loc=head[1]

        homedir=os.getcwd()
        if not os.path.exists(path_loc):
            os.mkdir(path_loc)
            self.results_window.append(path_loc+'/'+' is created.')
        else:
            self.results_window.append(path_loc+'/'+' already exists.')

        status=0
        currentdirlist=os.listdir(path_loc)
        if file_loc in currentdirlist:
            self.results_window.append(file_loc+' is found in local machine.')
            status = 1
        else:
            self.results_window.append(file_loc+' is NOT found in local machine.. start downloading.')

            # sftp
            import base64
            import pysftp
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            # sftp

            if self.tbox_host.toPlainText()!=[] and self.tbox_user.toPlainText()!=[] and self.tbox_pword.toPlainText()!=[] and self.tbox_basepath.toPlainText()!=[] and self.tbox_targetpath.toPlainText()!=[]:
                host=str(self.tbox_host.toPlainText())
                username=str(self.tbox_user.toPlainText())
                password=str(self.tbox_pword.toPlainText())
                basepath=str(self.tbox_basepath.toPlainText())
                targetpath=str(self.tbox_targetpath.toPlainText())

                self.tbox_selected_download.setText(file_loc)
                with pysftp.Connection(host=host, username=username, password=base64.b64decode(password),cnopts=cnopts) as sftp:
                    if sftp.exists(basepath):
                        with sftp.cd(basepath):
                            if sftp.exists(targetpath):
                                with sftp.cd(targetpath):
                                        if sftp.exists(path_loc):
                                            with sftp.cd(path_loc):
                                                if sftp.exists(file_loc):
                                                    os.chdir(path_loc)
                                                    sftp.get(file_loc, preserve_mtime=True,callback=self.printProgress)
                                                    self.results_window.append(file_loc+' download completed.')
                                                    os.chdir(homedir)
                                                    status=2
                                                else:
                                                    self.results_window.append(file_loc+' is not found in '+host)
                                                    status=-1
                                        else:
                                            self.results_window.append(path_loc+' is not found in '+host)
                                            status=-2

                            else:
                                self.results_window.append(targetpath+' is not found in '+host)
                                status=-3
                    else:
                        self.results_window.append(basepath+' is not found in '+host)
                        status=-4

        self.results_window.append('file check is completed with status = '+str(status))
        return status
        ## check cluster

    def import_multdim_comps(self,filename,withghost=0):
        try:
            filename
        except NameError:
            self.results_window.append(filename+' is NOT found')
        else:
            self.results_window.append(filename+' is found')
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
            self.results_window.append('cogent_time ='+str(cogent_time))
    
            ghost = File['level_0']['data_attributes'].attrs.get('ghost')
            self.results_window.append('ghost='+str(ghost))
            comps = File['level_0']['data_attributes'].attrs.get('comps')
            self.results_window.append('cogent_time ='+str(comps))
        
            boxes=File['level_0']['boxes'][:]
            num_decomposition = len(boxes)#boxes.shape[0]
            
            dim=len(boxes[0])/2
            
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
        
            print 'dim=',dim
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
            print 'num_cell_loc_with_ghost_tuple=',num_cell_loc_with_ghost_tuple
    
    
    
         
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
         
            for i in range(num_decomposition):
                cells = File['level_0']['boxes'][i]
                sys.stdout.write('.')
                sys.stdout.flush()
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
    
            
    
    
            print ' collected ', num_decomposition, 'decompositions'
            if (withghost==1):
                print 'Added outer ghost cells',ghost
                print num_cell_total_comps,'->',dataNd_bvec_with_outer_ghost_comps.shape
                return dataNd_bvec_with_outer_ghost_comps, ghost, cogent_time
            elif (withghost==2):
                print 'Added inner and outer ghost cells',ghost
                print num_cell_total_comps,'->',dataNd_bvec_with_ghost_comps.shape
                return dataNd_bvec_with_ghost_comps, ghost, cogent_time
            else:
                print num_cell_total_comps,'->',dataNd_bvec_comps.shape
                return dataNd_bvec_comps, cogent_time

    def printProgress(self, transferred, toBeTransferred):
        #print "Transferred: {:5.2f} %\r".format(float(transferred)/toBeTransferred*100)
        #self.results_window.append("Transferred: {:5.2f} %".format(float(transferred)/toBeTransferred*100))
        self.pb_download.setValue(float(transferred)/toBeTransferred*100)

    def print_target_file_lines(self):
        target_file = (self.target_list_box.toPlainText())
        with open(target_file,'r') as fo:
            self.results_window.append('target file open')
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
        self.tbox_pword.setText(pword)
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

    def plotFigure(self):

        selected_items=self.lw_target.selectedItems()
        selected_files=[]
        for i in range(len(selected_items)):
            selected_files.append(str(selected_items[i].text()))

        for i in range(len(selected_files)):
            self.results_window.append(selected_files[i])

            status=0
            #from plot_cogent_pack import check_and_try_cluster
            status=self.check_and_try_cluster(selected_files[i])

            if status>0:
                if 'dfn' in selected_files[i]:
                    if 'hydrogen' in selected_files[i]:
                        speciesname='i'
                    elif 'electron' in selected_files[i]:
                        speciesname='e'
                    else:
                        speciesname='s'
                    x_pt=float(self.te_x_pt.toPlainText())
                    y_pt=float(self.te_y_pt.toPlainText())
                    z_pt=float(self.te_z_pt.toPlainText())
                    plot_output=self.tbox_localout.toPlainText()
                    from plot_cogent_pack import plot_dfn
                    plot_dfn(selected_files[i],speciesname=speciesname,x_pt=x_pt,y_pt=y_pt,z_pt=z_pt,targetdir=plot_output)
                if 'potential' in selected_files[i]:
                    plot_potential(selected_files[i],ghost=0,x_slice=0.5,y_slice=0.5,z_slice=0.5,targetdir=plot_output)
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
                print 'No', selected_files[i], 'is found.. skipping the file.'


    












if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


