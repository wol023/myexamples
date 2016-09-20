import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1 
from scipy import interpolate
from scipy import ndimage

import pylab as pl

#for least squre fit
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

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

# label formatter
import matplotlib.ticker as ticker 

#for sleep
import time

#for sftp
import base64
import pysftp
import os
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None





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

def printProgress(transferred, toBeTransferred):
    #print "Transferred: {0}\tOut of : {1}".format(transferred, toBeTransferred)
    print "Transferred: {:5.2f} %\r".format(float(transferred)/toBeTransferred*100),


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    #float_str = "{0:.6g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return r"$%.2g$"%f

def add_colorbar(im, aspect=20, pad_fraction=0.5, field=[], **kwargs):
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
    labels_math=[latex_float(i) for i in labels]
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels_math)
    
    cb.update_ticks()
    return cb


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
        print filename,"is NOT found."
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

def check_and_try_cluster(pathfilename,host=[],username=[],password=[],basepath=[],targetpath=[]):
    head=os.path.split(pathfilename)
    path_loc=head[0]
    file_loc=head[1]
    print path_loc
    print file_loc

    homedir=os.getcwd()

    if not os.path.exists(path_loc):
        os.mkdir(path_loc)
        print path_loc+'/', 'is created.'
    else:
        print path_loc+'/', 'already exists.'

    status=0

    currentdirlist=os.listdir(path_loc)
    if file_loc in currentdirlist:
        print file_loc, 'is found in local machine.'
        status = 1
    else:
        print file_loc, 'is NOT found in local machine.. start downloading.'
        if host!=[] and username!=[] and password!=[] and basepath!=[] and targetpath!=[]:
            with pysftp.Connection(host=host, username=username, password=base64.b64decode(password),cnopts=cnopts) as sftp:
                if sftp.exists(basepath):
                    with sftp.cd(basepath):
                        if sftp.exists(targetpath):
                            with sftp.cd(targetpath):
                                    if sftp.exists(path_loc):
                                        with sftp.cd(path_loc):
                                            if sftp.exists(file_loc):
                                                os.chdir(path_loc)
                                                sftp.get(file_loc, preserve_mtime=True,callback=printProgress)
                                                print file_loc,'download completed.'
                                                os.chdir(homedir)
                                                status=2
                                            else:
                                                print file_loc,'is not found in', host
                                                status=-1
                                    else:
                                        print path_loc,'is not found in', host
                                        status=-2

                        else:
                            print targetpath,'is not found in', host
                            status=-3
                else:
                    print basepath,'is not found in', host
                    status=-4

    return status
















########################################
import numpy as np
from mayavi import mlab

#import multi component variables 

def plot_Nd(var,ghostIn=[],title='',xlabel='xlabel',ylabel='ylabel',xaxis=[],wh=1,fig_size_x=800,fig_size_y=600,sliced=0,x_slice=-1,y_slice=-1,z_slice=-1,interpolation='none',label=''):
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
            print 'x_slice=', x_slice, type(x_slice)
            x_slice_pt=float(bounds_var_ol[2*0])+x_slice-1
        else: #considier it a number between 0.0 to 1.0
            x_slice_pt=float(bounds_var_ol[2*0])+(float(bounds_var_ol[2*0+1])-float(bounds_var_ol[2*0]))*x_slice-1
    if var_dim>1:
        if y_slice==-1:#default slice middle
            y_slice_pt=float(bounds_var_ol[2*1+1])/2+float(bounds_var_ol[2*1])/2-1
        else:
            if type(y_slice)==type(1):#consider it as point
                print 'y_slice=', y_slice, type(y_slice)
                y_slice_pt=float(bounds_var_ol[2*1])+y_slice-1
            else: #considier it a number between 0.0 to 1.0
                y_slice_pt=float(bounds_var_ol[2*1])+(float(bounds_var_ol[2*1+1])-float(bounds_var_ol[2*1]))*y_slice-1
    if var_dim>2:
        if z_slice==-1:#default slice middle
            z_slice_pt=float(bounds_var_ol[2*2+1])/2+float(bounds_var_ol[2*2])/2-1
        else:
            if type(z_slice)==type(1):#consider it as point
                print 'z_slice=', z_slice, type(z_slice)
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
            module_manager.vector_lut_manager.scalar_bar.title = title
            module_manager.vector_lut_manager.scalar_bar_representation.position2 = np.array([ 0.1,  0.8])
            module_manager.vector_lut_manager.scalar_bar_representation.position = np.array([ 0.05,  0.1])
            module_manager.vector_lut_manager.label_text_property.color = (1-wh,1-wh, 1-wh)
            module_manager.vector_lut_manager.title_text_property.color = (1-wh, 1-wh, 1-wh)


        elif sliced==0 and (x_slice==-1 and y_slice==-1 and z_slice==-1):
            #try iso plot 
            fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
            ch=mlab.contour3d(var[:,:,:,0],contours=10,transparent=True,opacity=0.8)
            cb=mlab.colorbar(title=title,orientation='vertical' )
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

            cb=mlab.colorbar(title=title,orientation='vertical' )
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
    elif var_dim==2:
        #possibly vpar mu plot, only plot first component
        init_plotting()
        fig=plt.figure()
        plt.subplot(111)
        #plt.gca().margins(0.1, 0.1)
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
        add_colorbar(im,field=var[:,:,0])
        plt.tight_layout()
        return fig
    elif var_dim==1:
        #simple line out plot
        print 'Try using oplot_1d'
        init_plotting()
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
        plt.tight_layout()
        return fig


    return 



def oplot_1d(var,fig=[],ghostIn=[],title='',xlabel='xlabel',ylabel='ylabel',xaxis=[],wh=1,fig_size_x=800,fig_size_y=600,linewidth=1.5,linestyle='-',color='b',label='',legend=[]):
    #consider vpar-f simple plot 
    if fig==[]:
        fig=plt.figure()
    else:
        fig=plt.figure(fig.number)
    ax1=plt.gca()
    if xaxis==[]:
        xaxis=np.linspace(-1,1,len(var))
    ax1.plot(xaxis,var,linewidth=linewidth,linestyle=linestyle,color=color,label=label)
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
    if legend==[]:
        pass
    else:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim([ymin, ymax*1.2])

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0,1.0))
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    plt.tight_layout()
    return fig

   

def get_vpar_mu_scales(num_cell_total_comps_tuple,Vpar_max=1.0,Mu_max=1.0):
    #num_cell_total_comps_tuple[-3] = vpar cell
    #num_cell_total_comps_tuple[-2] = mu cell
    vpar_cell_dim_begin = -1.0*Vpar_max+(Vpar_max*1.0+Vpar_max*1.0)/float(num_cell_total_comps_tuple[-3])/2.0
    vpar_cell_dim_end   =  1.0*Vpar_max-(Vpar_max*1.0+Vpar_max*1.0)/float(num_cell_total_comps_tuple[-3])/2.0
    mu_cell_dim_begin = Mu_max*0.0+(Mu_max*1.0-Mu_max*0.0)/float(num_cell_total_comps_tuple[-2])/2.0
    mu_cell_dim_end   = Mu_max*1.0-(Mu_max*1.0-Mu_max*0.0)/float(num_cell_total_comps_tuple[-2])/2.0
    VPAR_CELL,MU_CELL = np.mgrid[vpar_cell_dim_begin:vpar_cell_dim_end:(num_cell_total_comps_tuple[-3]*1j),mu_cell_dim_begin:mu_cell_dim_end:(num_cell_total_comps_tuple[-2]*1j)]
    
    #VPAR_SCALE = VPAR_CELL[:,0]*np.sqrt(mhat) #for trunk
    VPAR_SCALE = VPAR_CELL[:,0] #for mass dependent normalization
    MU_SCALE = MU_CELL[0,:]
    return VPAR_SCALE, MU_SCALE

def get_vpar_mu_scales_meshgrid(num_cell_total_comps_tuple,Vpar_max=1.0,Mu_max=1.0):
    #num_cell_total_comps_tuple[-3] = vpar cell
    #num_cell_total_comps_tuple[-2] = mu cell
    vpar_cell_dim_begin = -1.0*Vpar_max+(Vpar_max*1.0+Vpar_max*1.0)/float(num_cell_total_comps_tuple[-3])/2.0
    vpar_cell_dim_end   =  1.0*Vpar_max-(Vpar_max*1.0+Vpar_max*1.0)/float(num_cell_total_comps_tuple[-3])/2.0
    mu_cell_dim_begin = Mu_max*0.0+(Mu_max*1.0-Mu_max*0.0)/float(num_cell_total_comps_tuple[-2])/2.0
    mu_cell_dim_end   = Mu_max*1.0-(Mu_max*1.0-Mu_max*0.0)/float(num_cell_total_comps_tuple[-2])/2.0
    VPAR_CELL,MU_CELL = np.mgrid[vpar_cell_dim_begin:vpar_cell_dim_end:(num_cell_total_comps_tuple[-3]*1j),mu_cell_dim_begin:mu_cell_dim_end:(num_cell_total_comps_tuple[-2]*1j)]
    
    return VPAR_CELL, MU_CELL


def get_maxwellian_coef(dfnfilename,mi,ti,me,te,nhat):
    if 'electron' in dfnfilename:
        mhat = me
        That = te
        coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)
        print 'electron (coef, mhat, That) = ', coef_maxwell, mhat, That
        return coef_maxwell,mhat,That
    elif 'hydrogen' in dfnfilename:
        mhat = mi
        That = ti
        coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)
        print 'hydrogen (coef, mhat, That) = ', coef_maxwell, mhat, That
        return coef_maxwell,mhat,That
    else:
        mhat = 1.0
        That = 1.0
        coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)
        print 'default (coef, mhat, That) = ', coef_maxwell, mhat, That
        return coef_maxwell,mhat,That


def get_maxwellian_fitting(coef_maxwell,mhat,That,Bhat,data_dfn,VPAR_SCALE,MU_SCALE,mu_ind=0):
    #least square fitting on a slice of MU index=0
    guess_den =coef_maxwell 
    guess_temp =1.0/(2.0*That)
    guess_shift =0.0
    optimize_func = lambda z: z[0]*np.exp(-z[1]*(VPAR_SCALE-z[2])**2)-data_dfn[:,mu_ind,0] 
    
    est_den, est_temp, est_shift = leastsq(optimize_func, [guess_den, guess_temp, guess_shift])[0]
    fitted_f = est_den*np.exp(-est_temp*(VPAR_SCALE-est_shift)**2)
    t_fit = 1.0/(est_temp*2.0)
    n_fit = est_den*np.sqrt(np.pi)/(0.5*mhat/t_fit)**(1.5)/np.exp(-MU_SCALE[mu_ind]*Bhat/2.0/t_fit)
    vshift_fit = est_shift
    print '(n_fit, t_fit, vshift_fit)= ',n_fit,t_fit,vshift_fit
    return fitted_f, n_fit, t_fit, vshift_fit

def get_maxwellian_fitting_with_fixed_max(mhat,That,Bhat,data_dfn,VPAR_SCALE,MU_SCALE,mu_ind=0):
    #least square fitting on a slice of MU index=0
    guess_temp =1.0/(2.0*That)
    guess_shift =0.0
    fixed_max_den=max(data_dfn[:,mu_ind,0])
    optimize_func = lambda z: fixed_max_den*np.exp(-z[0]*(VPAR_SCALE-z[1])**2)-data_dfn[:,mu_ind,0] 
    
    est_temp, est_shift = leastsq(optimize_func, [guess_temp, guess_shift])[0]
    fitted_f = fixed_max_den*np.exp(-est_temp*(VPAR_SCALE-est_shift)**2)
    t_fit = 1.0/(est_temp*2.0)
    n_fit = fixed_max_den*np.sqrt(np.pi)/(0.5*mhat/t_fit)**(1.5)/np.exp(-MU_SCALE[mu_ind]*Bhat/2.0/t_fit)
    vshift_fit = est_shift
    print '(n_fit, t_fit, vshift_fit)=',n_fit,t_fit,vshift_fit
    return fitted_f, n_fit, t_fit, vshift_fit




def get_summation_over_velocity(dataNd_dfn_comps,Vpar_max,Mu_max):
    num_cell_total_comps_tuple=dataNd_dfn_comps.shape
    #num_cell_total_comps_tuple[-6] = x cell
    #num_cell_total_comps_tuple[-5] = y cell ; x for 4D
    #num_cell_total_comps_tuple[-4] = z cell ; y for 4D
    #num_cell_total_comps_tuple[-3] = vpar cell
    #num_cell_total_comps_tuple[-2] = mu cell
    #num_cell_total_comps_tuple[-1] = comps
    dim=len(num_cell_total_comps_tuple)-1
    print 'dim=',dim

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
        for d in range(num_compcell):
            for i in range(num_xcell):
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
    print f_vpar_mu_sum.shape
    return f_vpar_mu_sum

####sum over mu
###f_mu_sum = np.zeros(num_vparcell)
###if dim<5:
###    for i in range(num_vparcell):
###        for j in range(num_mucell):
###            f_mu_sum[i]=f_mu_sum[i]+dataNd[x_pt,y_pt,i,j]
###        f_mu_sum[i]=f_mu_sum[i]/num_mucell
###else:
###    for i in range(num_vparcell):
###        for j in range(num_mucell):
###            f_mu_sum[i]=f_mu_sum[i]+dataNd[x_pt,y_pt,z_pt,i,j]
###        f_mu_sum[i]=f_mu_sum[i]/num_mucell
###



def plot_dfn(pathfilename,saveplots=1,showplots=0,x_pt=1,y_pt=1,z_pt=1,targetdir=[]):
    head=os.path.split(pathfilename)
    path=head[0]
    filename=head[1]
    print path
    print filename
    basedir=os.getcwd()
    if targetdir==[]:
        targetdir='./python_auto_plots'

    dataNd_dfn_comps=import_multdim_comps(filename=pathfilename)
    title_var=r'$f(\bar{v}_\parallel, \bar{\mu})$'
    num_cell_total_comps_tuple=dataNd_dfn_comps.shape
    from read_input_deck import *
    VPAR_SCALE, MU_SCALE = get_vpar_mu_scales(num_cell_total_comps_tuple,Vpar_max,Mu_max)
    VPAR_N, MU_N = get_vpar_mu_scales(num_cell_total_comps_tuple) #default max
    
    #velocity space
    fig_dfn2d=plot_Nd(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],title=title_var)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        plt.savefig(filename.replace('.hdf5','.vpar_mu.png'))
        plt.savefig(filename.replace('.hdf5','.vpar_mu.eps'))
        os.chdir(basedir)
    if showplots==0:
        plt.close('all')
    
    fig_dfn2d_interp=plot_Nd(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],title=title_var,interpolation='spline36')
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        plt.savefig(filename.replace('.hdf5','.vpar_mu_interp.png'))
        plt.savefig(filename.replace('.hdf5','.vpar_mu_interp.eps'))
        os.chdir(basedir)
    if showplots==0:
        plt.close('all')
        #plt.close(fig_dfn2d)



#    #example vpar plot overplot
#    fig=oplot_1d(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,0,:],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,0,:])),label='COGENT'%MU_N[0],legend=1 )
#    #oplot_1d(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,1,:],fig=fig,xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,1,:])),label='f(mu=%g)'%(MU_N[1]),legend=1 )
#    #example mu=0 fitting
#    coef_maxwell,mhat,That = get_maxwellian_coef(pathfilename,ion_mass,t0_grid_func,elec_mass,et0_grid_func,nhat)
#    fitted_f, n_fit, t_fit, vshift_fit = get_maxwellian_fitting(coef_maxwell,mhat, That,Bhat,dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],VPAR_SCALE,MU_SCALE,mu_ind=0)
#    # plot fitting
#    legend_maxwellian = '\n\nMAXWELLIAN\n'+r'$n, T, V_s$'+' = (%.1f, %.1f, %.1g)'%(n_fit,t_fit,vshift_fit)
#    oplot_1d(fitted_f,fig,title='',linewidth=1.5, linestyle='--',color='k',label=legend_maxwellian,legend=1)
#    if saveplots>0:
#        if not os.path.exists(targetdir):
#            os.mkdir(targetdir)
#            print targetdir+'/', 'is created.'
#        else:
#            print targetdir+'/', 'already exists.'
#        os.chdir(targetdir)
#        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_compare.png'))
#        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_compare.eps'))
#        os.chdir(basedir)
#    if showplots==0:
#        plt.close('all')
#
#    #maxwell difference plot
#    fig=oplot_1d(fitted_f-dataNd_dfn_comps[x_pt,y_pt,z_pt,:,0,0],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,0,0])),label='(f_M-f_S)/f_M'%MU_N[0],legend=1 )
#    if saveplots>0:
#        if not os.path.exists(targetdir):
#            os.mkdir(targetdir)
#            print targetdir+'/', 'is created.'
#        else:
#            print targetdir+'/', 'already exists.'
#        os.chdir(targetdir)
#        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_diff.png'))
#        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_diff.eps'))
#        os.chdir(basedir)
#    if showplots==0:
#        plt.close('all')

    # For all mu maxwellian fitting
    coef_maxwell,mhat,That = get_maxwellian_coef(pathfilename,ion_mass,t0_grid_func,elec_mass,et0_grid_func,nhat)
    data_2d_shape=dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:].shape
    data2d_maxwell=np.linspace(0.0,0.1,data_2d_shape[0]*data_2d_shape[1]*data_2d_shape[2]).reshape((data_2d_shape[0],data_2d_shape[1],data_2d_shape[2]))
    n_fit=np.linspace(0.0,0.1,data_2d_shape[1])
    t_fit=np.linspace(0.0,0.1,data_2d_shape[1])
    vshift_fit=np.linspace(0.0,0.1,data_2d_shape[1])
    for i in range(data_2d_shape[1]):
        sys.stdout.write('mu_ind='+str(i))
        sys.stdout.flush()
        #fitted_f, n_fit[i], t_fit[i], vshift_fit[i] = get_maxwellian_fitting(coef_maxwell,mhat,That,Bhat,dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],VPAR_SCALE,MU_SCALE,mu_ind=i)
        fitted_f, n_fit[i], t_fit[i], vshift_fit[i] = get_maxwellian_fitting_with_fixed_max(mhat,That,Bhat,dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],VPAR_SCALE,MU_SCALE,mu_ind=i)
        data2d_maxwell[:,i,0]=fitted_f

    #
    #VPAR_CELL,MU_CELL=get_vpar_mu_scales_meshgrid(num_cell_total_comps_tuple,Vpar_max,Mu_max)


    plot_Nd(data2d_maxwell-dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],title="$f_M - f_{COGENT}$",interpolation='spline36')
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_2d_diff_interp.png'))
        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_2d_diff_interp.eps'))
        os.chdir(basedir)
    if showplots==0:
        plt.close('all')

    #plot difference
    mu_pt=1

    fig=oplot_1d(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,:],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,:])),label='COGENT'%MU_N[mu_pt],legend=1 )
    legend_maxwellian = 'MAXWELLIAN '+r'$n, T, V_s$'+' = (%.1f, %.1f, %.1g)'%(n_fit[mu_pt],t_fit[mu_pt],vshift_fit[mu_pt])
    oplot_1d(data2d_maxwell[:,mu_pt,0],fig,title='',linewidth=1.5, linestyle='--',color='k',label=legend_maxwellian,legend=1,ylabel='f_M, f_S')
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_2d_mu_1.png'))
        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_2d_mu_1.eps'))
        os.chdir(basedir)
    if showplots==0:
        plt.close('all')


     #maxwell difference plot
    fig=oplot_1d(data2d_maxwell[:,mu_pt,0]-dataNd_dfn_comps[x_pt,y_pt,z_pt,:,mu_pt,0],xaxis=np.linspace(-1,1,len(data2d_maxwell[:,mu_pt,0])),label='(mu=%g)'%MU_N[mu_pt],legend=1,ylabel='f_M - f_S' )
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_2d_diff_at_1.png'))
        plt.savefig(filename.replace('.hdf5','.vpar_maxwell_2d_diff_at_1.eps'))
        os.chdir(basedir)
    if showplots==0:
        plt.close('all')

    

    
    #get density by summation over velocity
    f_vpar_mu_sum = get_summation_over_velocity(dataNd_dfn_comps,Vpar_max,Mu_max)
    fig=plot_Nd(f_vpar_mu_sum,title='integrated f')
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.f_sum_mlab_iso.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.f_sum_mlab_iso.eps'))
        time.sleep(1)
        os.chdir(basedir)
    #fig.scene.save_ps('fig1_mlab_iso.pdf')
    #mlab.show()
    #arr=mlab.screenshot()
    #plt.imshow(arr)
    #plt.axis('off')
    #plt.savefig('fig1_mlab_iso.eps')
    #plt.savefig('fig1_mlab_iso.pdf')
    #fig.scene.close()
    if showplots==0:
        mlab.close(all=True)

    #sliced plot
    fig=plot_Nd(f_vpar_mu_sum,title='integrated f',x_slice=x_pt,y_slice=y_pt,z_slice=z_pt)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.f_sum_mlab_slice.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.f_sum_mlab_slice.eps'))
        time.sleep(1)
        os.chdir(basedir)
    if showplots==0:
        mlab.close(all=True)

    return
 

def plot_potential(pathfilename,saveplots=1,showplots=0,ghost=0,x_slice=0.5,y_slice=0.5,z_slice=0.5,targetdir=[]):
    head=os.path.split(pathfilename)
    path=head[0]
    filename=head[1]
    print path
    print filename
    basedir=os.getcwd()
    if targetdir==[]:
        targetdir='./python_auto_plots'
    
    dataNd_potential_comps=import_multdim_comps(filename=pathfilename)
    if ghost>0:
        dataNd_potential_with_outer_ghost_comps,num_ghost_potential=import_multdim_comps(filename=pathfilename,withghost=1)
    title_var='potential'
    
    fig=plot_Nd(dataNd_potential_comps,title=title_var)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.potential_mlab_iso.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.potential_mlab_iso.eps'))
        time.sleep(1)
        os.chdir(basedir)
    if showplots==0:
        mlab.close(all=True)
    fig=plot_Nd(dataNd_potential_comps,title=title_var,sliced=1,x_slice=x_slice,y_slice=y_slice,z_slice=z_slice)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.potential_mlab_slice.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.potential_mlab_slice.eps'))
        time.sleep(1)
        os.chdir(basedir)
    if showplots==0:
        mlab.close(all=True)



    if ghost>0:
        fig=plot_Nd(dataNd_potential_with_outer_ghost_comps,num_ghost_potential,title=title_var)
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
                print targetdir+'/', 'is created.'
            else:
                print targetdir+'/', 'already exists.'
            os.chdir(targetdir)
            mlab.savefig(filename.replace('.hdf5','.potential_mlab_ghost_iso.png'))
            fig.scene.save_ps(filename.replace('.hdf5','.potential_mlab_ghost_iso.eps'))
            time.sleep(1)
            os.chdir(basedir)
        if showplots==0:
            mlab.close(all=True)
        fig=plot_Nd(dataNd_potential_with_outer_ghost_comps,num_ghost_potential,title=title_var,x_slice=x_slice,y_slice=y_slice,z_slice=z_slice)
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
                print targetdir+'/', 'is created.'
            else:
                print targetdir+'/', 'already exists.'
            os.chdir(targetdir)
            mlab.savefig(filename.replace('.hdf5','.potential_mlab_ghost_slice.png'))
            fig.scene.save_ps(filename.replace('.hdf5','.potential_mlab_ghost_slice.eps'))
            time.sleep(1)
            os.chdir(basedir)
        if showplots==0:
            mlab.close(all=True)

    return

    
def plot_bvec(pathfilename,saveplots=1,showplots=0,ghost=0,targetdir=[]):
    head=os.path.split(pathfilename)
    path=head[0]
    filename=head[1]
    print path
    print filename
    basedir=os.getcwd()
    if targetdir==[]:
        targetdir='./python_auto_plots'

    dataNd_bvec_comps=import_multdim_comps(filename=pathfilename)
    if ghost>0:
        dataNd_bvec_with_outer_ghost_comps,num_ghost_bvec =import_multdim_comps(filename=pathfilename,withghost=1)
    title_var='B field'
    
    fig=plot_Nd(dataNd_bvec_comps,title=title_var)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.bvec_mlab.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.bvec_mlab_ps.eps'))
        #fig.scene.save_gl2ps(filename.replace('.hdf5','.bvec_mlab_gl2ps.eps'))
        time.sleep(1)
        os.chdir(basedir)
    if showplots==0:
        mlab.close(all=True)

    if ghost>0:
        fig=plot_Nd(dataNd_bvec_with_outer_ghost_comps,num_ghost_bvec,title=title_var) 
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
                print targetdir+'/', 'is created.'
            else:
                print targetdir+'/', 'already exists.'
            os.chdir(targetdir)
            mlab.savefig(filename.replace('.hdf5','.bvec_mlab_ghost.png'))
            fig.scene.save_ps(filename.replace('.hdf5','.bvec_mlab_ghost_ps.eps'))
            #fig.scene.save_gl2ps(filename.replace('.hdf5','.bvec_mlab_ghost_gl2ps.eps'))
            time.sleep(1)
            os.chdir(basedir)
        if showplots==0:
            mlab.close(all=True)

    return

    
def plot_evec(pathfilename,saveplots=1,showplots=0,ghost=0,targetdir=[]):
    head=os.path.split(pathfilename)
    path=head[0]
    filename=head[1]
    print path
    print filename
    basedir=os.getcwd()
    if targetdir==[]:
        targetdir='./python_auto_plots'

    dataNd_evec_comps=import_multdim_comps(filename=pathfilename)
    if ghost>0:
        dataNd_evec_with_outer_ghost_comps,num_ghost_evec=import_multdim_comps(filename=pathfilename,withghost=1)
    title_var='E field'
    
    fig=plot_Nd(dataNd_evec_comps,title=title_var)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.evec_mlab.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.evec_mlab_ps.eps'))
        #fig.scene.save_gl2ps(filename.replace('.hdf5','.evec_mlab_gl2ps.eps'))
        time.sleep(1)
        os.chdir(basedir)
    if showplots==0:
        mlab.close(all=True)

    if ghost>0:
        fig=plot_Nd(dataNd_evec_with_outer_ghost_comps,num_ghost_evec,title=title_var)
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
                print targetdir+'/', 'is created.'
            else:
                print targetdir+'/', 'already exists.'
            os.chdir(targetdir)
            mlab.savefig(filename.replace('.hdf5','.evec_mlab_ghost.png'))
            fig.scene.save_ps(filename.replace('.hdf5','.evec_mlab_ghost_ps.eps'))
            #fig.scene.save_gl2ps(filename.replace('.hdf5','.evec_mlab_ghost_gl2ps.eps'))
            time.sleep(1)
            os.chdir(basedir)
        if showplots==0:
            mlab.close(all=True)

    return


    
def plot_density(pathfilename,saveplots=1,showplots=0,ghost=0,x_slice=0.5,y_slice=0.5,z_slice=0.5,targetdir=[]):
    head=os.path.split(pathfilename)
    path=head[0]
    filename=head[1]
    print path
    print filename
    basedir=os.getcwd()
    if targetdir==[]:
        targetdir='./python_auto_plots'

    dataNd_density_comps=import_multdim_comps(filename=pathfilename)
    if ghost>0:
        dataNd_density_with_outer_ghost_comps,num_ghost_density=import_multdim_comps(filename=pathfilename,withghost=1)
    title_var='density'
    
    fig=plot_Nd(dataNd_density_comps,title=title_var)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.mlab_iso.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.mlab_iso.eps'))
        time.sleep(1)
        os.chdir(basedir)
    if showplots==0:
        mlab.close(all=True)

    fig=plot_Nd(dataNd_density_comps,title=title_var,sliced=1,x_slice=x_slice,y_slice=y_slice,z_slice=z_slice)
    if saveplots>0:
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
            print targetdir+'/', 'is created.'
        else:
            print targetdir+'/', 'already exists.'
        os.chdir(targetdir)
        mlab.savefig(filename.replace('.hdf5','.mlab_slice.png'))
        fig.scene.save_ps(filename.replace('.hdf5','.mlab_slice.eps'))
        time.sleep(1)
        os.chdir(basedir)
    if showplots==0:
        mlab.close(all=True)

    if ghost>0:
        fig=plot_Nd(dataNd_density_with_outer_ghost_comps,num_ghost_density,title=title_var)
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
                print targetdir+'/', 'is created.'
            else:
                print targetdir+'/', 'already exists.'
            os.chdir(targetdir)
            mlab.savefig(filename.replace('.hdf5','.mlab_ghost_iso.png'))
            fig.scene.save_ps(filename.replace('.hdf5','.mlab_ghost_iso.eps'))
            time.sleep(1)
            os.chdir(basedir)
        if showplots==0:
            mlab.close(all=True)

        fig=plot_Nd(dataNd_density_with_outer_ghost_comps,num_ghost_density,title=title_var,x_slice=x_slice,y_slice=y_slice,z_slice=z_slice)
        if saveplots>0:
            if not os.path.exists(targetdir):
                os.mkdir(targetdir)
                print targetdir+'/', 'is created.'
            else:
                print targetdir+'/', 'already exists.'
            os.chdir(targetdir)
            mlab.savefig(filename.replace('.hdf5','.mlab_ghost_slice.png'))
            fig.scene.save_ps(filename.replace('.hdf5','.mlab_ghost_slice.eps'))
            time.sleep(1)
            os.chdir(basedir)
        if showplots==0:
            mlab.close(all=True)

    return









