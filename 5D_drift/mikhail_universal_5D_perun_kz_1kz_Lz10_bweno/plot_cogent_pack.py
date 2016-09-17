import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1 
from scipy import interpolate
from scipy import ndimage

import pylab as pl

#for least squre fit
from scipy.optimize import leastsq

#to 3D plot
from mayavi import mlab
import mayavi

#to find kinetic.in
import os, fnmatch
import ConfigParser

#for stdout print
import sys 

#To parse n0_grid_func
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy import *
from sympy.abc import x, y, z
#from sympy.utilities.lambdify import implemented_function
#from sympy import Function

#setup plot
graphDPI =200
# set global settings
def init_plotting(form=''):
    if (form == '2x3'):
        plt.rcParams['figure.figsize'] = (16, 9)
    elif (form == '2x2'):
        plt.rcParams['figure.figsize'] = (12, 9)
    else:
        plt.rcParams['figure.figsize'] = (4, 3)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 1.5*plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
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

#        plt.gca().spines['right'].set_color('none')
#        plt.gca().spines['top'].set_color('none')
#        plt.gca().xaxis.set_ticks_position('bottom')
#        plt.gca().yaxis.set_ticks_position('left')
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0/aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result

def findpertbation(inputstr,tofind):
    result = []
    ind_first=inputstr.find(tofind)
    ind_next=ind_first
    while ind_next >= 0:
        ind_check=inputstr[ind_next:].find(tofind)
        if ind_check>=0:
            ind_first=ind_next
            #print ind_next
            balance=0
            token = ''
            for letter in inputstr[ind_first:]:
                if ind_next==len(inputstr)-1:
                    ind_next=-1
                else:
                    ind_next+=1
                if letter =='(':
                    balance=balance+1
                    token +=letter
                    #print token,balance
                elif letter ==')':
                    balance=balance-1
                    if balance == 0:
                        token+=letter
                        #print token,balance
                        break
                    else:
                        token+=letter
                        #print token,balance
                else:
                    if balance<1:
                        continue
                    else:
                        token+=letter
                        #print token,balance
            #now token has something like (1*y)
            #replace pi to np.pi
            
            token=token.replace('pi','np.pi')
            token=token.replace('PI','np.pi')
            token=token.replace('Pi','np.pi')

            result.append(token)
        else:
            ind_next=-1

    return result

def findmodenumber(token,tofind): #token is (1*y), tofind is y
    modenumber=0.0
    replaced_token=''
    ind_var=token.find(tofind)
    ind_pivot=ind_var
    #scan to the left
    #print 'ind_var=',ind_var
    end_var=-1
    start_var=ind_var
    balance=0
    while ind_var-1>=0:
        #print 'token[',ind_var,']=',token[ind_var]
        if token[ind_var]==')':
            balance+=1
            ind_var-=1
            continue
        if token[ind_var]=='(':
            balance-=1
            if balance<1:
                ind_var-=1
            continue

        if token[ind_var]=='*':
            if end_var==-1:#end_var was not set because * was ommitted
                end_var=ind_var-1
            #print 'end_var=',end_var
            ind_var-=1
            continue
        #elif token[ind_var-1]=='(' or token[ind_var-1]=='+' or token[ind_var-1]=='-' or token[ind_var-1].isalpha():
        elif (token[ind_var-1]=='+' or token[ind_var-1]=='-') and balance<1:
            start_var=ind_var-1
            ind_var-=1
            #print 'start_var=',start_var
            #print 'ind_var=',ind_var
            break
        elif token[ind_var].isdigit()or token[ind_var]=='.' or ind_var==ind_pivot:#digits
            if end_var==-1 and token[ind_var-1]=='*':
                end_var=ind_var-1
            elif end_var==-1:
                end_var=ind_var
            ind_var-=1
            #print 'ind_var=',ind_var
            continue
        else:
            ind_var-=1
            continue
    #if start_var!=-1 and start_var!=end_var:
    #    print 'token[',start_var,':',end_var,']=',token[start_var:end_var]

    #scan to the right
    ind_var=ind_pivot
    balance=0
    #print 'ind_var=',ind_var
    while ind_var+1<=len(token):
        #print 'token[',ind_var,']=',token[ind_var]
        if token[ind_var]=='(':
            balance+=1
            #print 'token[',start_var,':',ind_var,']=',token[start_var:ind_var]
            #print 'token[',ind_var,']=',token[ind_var],':balance=',balance
            ind_var+=1
            continue
        if token[ind_var]==')':
            balance-=1
            #print 'token[',start_var,':',ind_var,']=',token[start_var:ind_var]
            #print 'token[',ind_var,']=',token[ind_var],':balance=',balance
            if balance<1 and balance>=0:
                end_var=ind_var+1
                ind_var+=1
                break
            elif balance<0:
                end_var=ind_var
                ind_var+=1
                break
            else:
                ind_var+=1
            continue

        if (token[ind_var]=='+' or token[ind_var]=='-') and balance<1:
            end_var=ind_var
            ind_var+=1
            break
        else:
            ind_var+=1

    if start_var!=-1 and start_var!=end_var:
        #print tofind,': token[',start_var,':',end_var,']=',token[start_var:end_var]
        replaced_token=token[start_var:end_var].replace(tofind,'1.0')
        #print tofind,': replaced_token=',replaced_token

    if len(replaced_token)>0:
        modenumber=eval(replaced_token)
        #print tofind,': evaluated_token=',modenumber
    return modenumber


############################################
import numpy as np
import h5py
#for stdout print
import sys 

#import multi component variables 
#withghost=0 : no ghost cells
#withghost=1 : include outer ghost cells
#withghost=2 : include inner and outer ghost cells

def import_multdim_comps(filename,withghost=0):
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
        print 'ghost=',ghost
        comps = File['level_0']['data_attributes'].attrs.get('comps')
        print 'comps=',comps
    
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
            return dataNd_bvec_with_outer_ghost_comps, ghost
        elif (withghost==2):
            print 'Added inner and outer ghost cells',ghost
            print num_cell_total_comps,'->',dataNd_bvec_with_ghost_comps.shape
            return dataNd_bvec_with_ghost_comps, ghost
        else:
            print num_cell_total_comps,'->',dataNd_bvec_comps.shape
            return dataNd_bvec_comps



########################################
import numpy as np
from mayavi import mlab

#import multi component variables 

def plot_Nd(var,ghostIn=[],titleIn='variable',wh=1,fig_size_x=800,fig_size_y=600,sliced=0,x_slice=-1,y_slice=-1,z_slice=-1):
    #wh=1 # 0: black background, 1: whithe background
    #first check the rank of input data
    var_shape=var.shape
    var_components=var_shape[-1]
    var_dim=len(var_shape)-1

    #set environment
    if ghostIn==[]:
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
        for i in range((len(var.shape)-1)):
            bounds_var_ol[2*i]=1
            bounds_var_ol[2*i+1]=var.shape[i]
        ax_line_width=1.0
        ol_line_width=1.0
 
    else:
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

    if x_slice<0:#default slice middle
        x_slice_pt=float(bounds_var_ol[2*0+1])/2+float(bounds_var_ol[2*0])/2-1
    else:
        x_slice_pt=float(bounds_var_ol[2*0])+(float(bounds_var_ol[2*0+1])-float(bounds_var_ol[2*0]))*x_slice-1
    if y_slice<0:
        y_slice_pt=float(bounds_var_ol[2*1+1])/2+float(bounds_var_ol[2*1])/2-1
    else:
        y_slice_pt=float(bounds_var_ol[2*1])+(float(bounds_var_ol[2*1+1])-float(bounds_var_ol[2*1]))*y_slice-1
    if z_slice<0:
        z_slice_pt=float(bounds_var_ol[2*2+1])/2+float(bounds_var_ol[2*2])/2-1
    else:
        z_slice_pt=float(bounds_var_ol[2*2])+(float(bounds_var_ol[2*2+1])-float(bounds_var_ol[2*2]))*z_slice-1

        print bounds_var_ax
        print bounds_var_ol
        print x_slice_pt,x_slice
        print y_slice_pt,y_slice
        print z_slice_pt,z_slice

 
    #Start plotting   
    if var_dim==3:
        #try 3D plot using mayavi
        if var_components>1:
            #try vector field plot (x,y,z) = 0 2 -1
            fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
            src=mlab.pipeline.vector_field(var[:,:,:,0],var[:,:,:,2],-var[:,:,:,1])
            vh=mlab.pipeline.vectors(src,mask_points=4)
            #engine=mlab.get_engine()
            #vectors = engine.scenes[0].children[0].children[0].children[0]
            #vectors.glyph.glyph_source.glyph_source = vectors.glyph.glyph_source.glyph_dict['arrow_source']
            vh.glyph.glyph_source.glyph_source = vh.glyph.glyph_source.glyph_dict['arrow_source']
            vh.glyph.glyph.scale_factor = 1.0

            engine=mlab.get_engine()
            s=engine.current_scene
            module_manager = s.children[0].children[0]
            module_manager.vector_lut_manager.show_scalar_bar = True
            module_manager.vector_lut_manager.show_legend = True
            module_manager.vector_lut_manager.scalar_bar.title = titleIn
            module_manager.vector_lut_manager.scalar_bar_representation.position2 = np.array([ 0.1,  0.8])
            module_manager.vector_lut_manager.scalar_bar_representation.position = np.array([ 0.05,  0.1])
            module_manager.vector_lut_manager.label_text_property.color = (1-wh,1-wh, 1-wh)
            module_manager.vector_lut_manager.title_text_property.color = (1-wh, 1-wh, 1-wh)

###3
#            cb=mlab.vectorbar(title=titleIn,orientation='vertical' )
            #cb.title_text_property.color=(1-wh,1-wh,1-wh)
            #cb.label_text_property.color=(1-wh,1-wh,1-wh)
#            engine=mlab.get_engine()
#            module_manager = engine.scenes[0].children[0].children[0]
#            module_manager.vector_lut_manager.scalar_bar.orientation = 'vertical'
#            module_manager.vector_lut_manager.scalar_bar_representation.position2 = np.array([ 0.1,  0.8])
#            module_manager.vector_lut_manager.scalar_bar_representation.position = np.array([ 0.05,  0.1])
#            module_manager.vector_lut_manager.scalar_bar_representation.maximum_size = np.array([100000, 100000])

        elif sliced==0 and (x_slice==-1 and y_slice==-1 and z_slice==-1):
            #try iso plot 
            fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
            ch=mlab.contour3d(var[:,:,:,0],contours=10,transparent=True,opacity=0.8)
            cb=mlab.colorbar(title=titleIn,orientation='vertical' )
            cb.title_text_property.color=(1-wh,1-wh,1-wh)
            cb.label_text_property.color=(1-wh,1-wh,1-wh)
        else:
            #try slice plot 
            fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
            if sliced>0:
                sxh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='x_axes',slice_index=x_slice_pt)
                sxh.ipw.slice_position=x_slice_pt+1
                syh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='y_axes',slice_index=y_slice_pt)
                syh.ipw.slice_position=y_slice_pt+1
                szh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='z_axes',slice_index=z_slice_pt)
                szh.ipw.slice_position=z_slice_pt+1
            else:
                if x_slice>-1:
                    sxh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='x_axes',slice_index=x_slice_pt)
                    sxh.ipw.slice_position=x_slice_pt+1
                if y_slice>-1:
                    syh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='y_axes',slice_index=y_slice_pt)
                    syh.ipw.slice_position=y_slice_pt+1
                if z_slice>-1:
                    szh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='z_axes',slice_index=z_slice_pt)
                    szh.ipw.slice_position=z_slice_pt+1

            cb=mlab.colorbar(title=titleIn,orientation='vertical' )
            cb.title_text_property.color=(1-wh,1-wh,1-wh)
            cb.label_text_property.color=(1-wh,1-wh,1-wh)


        ax = mlab.axes(nb_labels=5,ranges=range_var)
        ax.axes.property.color=(1-wh,1-wh,1-wh)
        ax.axes.property.line_width = ax_line_width
        ax.axes.bounds=(bounds_var_ax)
        ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
        ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
        ax.axes.label_format='%.2f'
        ol=mlab.outline(color=(1-wh,1-wh,1-wh),extent=bounds_var_ol)
        ol.actor.property.line_width = ol_line_width
        mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')
        return fig

    elif var_dim==5:
        if var_components>1:
            print 'dfn plot with mult components'
            return
        else:
            print 'dfn plot with single component'
            #try collect for density
            return 
    return 













