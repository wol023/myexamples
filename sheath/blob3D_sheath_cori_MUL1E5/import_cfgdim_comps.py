import numpy as np
import h5py
#for stdout print
import sys 

#import multi component variables 
#withghost=0 : no ghost cells
#withghost=1 : include outer ghost cells
#withghost=2 : include inner and outer ghost cells

def import_cfgdim_comps(filename,withghost=0):
    try:
        filename
    except NameError:
        print filename,"is NOTi found."
    else:
        print filename,"is found."
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
        
        ghost = File['level_0']['data_attributes'].attrs.get('ghost')
        #print 'ghost=',ghost
        comps = File['level_0']['data_attributes'].attrs.get('comps')
        #print 'comps=',comps
    
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
    
        if dim<3:
            dir_x=0
            dir_y=1
        else:
            dir_x=0
            dir_y=1
            dir_z=2
    
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
                
            if dim<3:
                dataNd_loc_with_ghost_comps=data[previous_index:prod_num_cell_loc_with_ghost*(i+1)].reshape((num_cell_loc_with_ghost[dir_x],num_cell_loc_with_ghost[dir_y]),order='F')
            else:
                for d in range(comps):
                    #print 'i=',i
                    #print 'd=',d
                    #print 'previous_index=',previous_index
                    #print 'loc_end_index =',prod_num_cell_loc_with_ghost*(d+1)+prod_num_cell_loc_with_ghost*(i)*comps
                    dataNd_loc_with_ghost_comps[:,:,:,d]=data[previous_index:prod_num_cell_loc_with_ghost*(d+1)+prod_num_cell_loc_with_ghost*(i)*comps].reshape((num_cell_loc_with_ghost[dir_x],num_cell_loc_with_ghost[dir_y],num_cell_loc_with_ghost[dir_z]),order='F') 
                    previous_index=prod_num_cell_loc_with_ghost*(d+1)+prod_num_cell_loc_with_ghost*(i)*comps
                    
        
            for d in range(comps):
                dataNd_bvec_with_ghost_comps[cells_shift_with_ghost[dir_x]:cells_shift_with_ghost[dim+dir_x]+1, cells_shift_with_ghost[dir_y]:cells_shift_with_ghost[dim+dir_y]+1, cells_shift_with_ghost[dir_z]:cells_shift_with_ghost[dim+dir_z]+1,d]=dataNd_loc_with_ghost_comps[:,:,:,d]
        
        
        File.close()
    
        #removing all ghost cells
        for i in range(num_cell_total_comps[0]):
            current_decomp_i=i/num_cell_loc[0]
            for j in range(num_cell_total[1]):
                current_decomp_j=j/num_cell_loc[1]
                for k in range(num_cell_total[2]):
                    current_decomp_k=k/num_cell_loc[2]
                    for d in range(comps):
                        dataNd_bvec_comps[i,j,k,d]=dataNd_bvec_with_ghost_comps[ghost[0]+ghost[0]*2*current_decomp_i+i,ghost[1]+ghost[1]*2*current_decomp_j+j,ghost[2]+ghost[2]*2*current_decomp_k+k,d]

    #removing inner ghost cells
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


        print ' collected ', num_decomposition, 'decompositions'
        if (withghost==1):
            print 'Added outer ghost cells',ghost
            print num_cell_total_comps,'->',dataNd_bvec_with_outer_ghost_comps.shape
            return dataNd_bvec_with_outer_ghost_comps, ghost
        elif (withghost==2):
            print 'Added inner and outer ghost cells',ghost
            print num_cell_total_comps,'->',dataNd_bvec_with_ghost_comps.shape
            return dataNd_bvec_with_ghost_comps, ghost
        else:
            print num_cell_total_comps,'->',dataNd_bvec_comps.shape
            return dataNd_bvec_comps


