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




#############################################################
########################## diagnostic points
x_pt = 4
y_pt = 5
z_pt = 6
#vpar_rbf_refine = 5
#mu_rbf_refine = 5
#rbf_smooth = 0
#############################################################
#############################################################
# import dfn
hdffilename='./plt_dfn_plots/plt.1.hydrogen.dfn0004.5d.hdf5'
# import potential
potentialfilename='./plt_potential_plots/plt.potential0001.3d.hdf5'











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




try:
    potentialfilename
except NameError:
    print "skip potential plots."
else:
    print "potential file is set."
    
    File =h5py.File(potentialfilename,'r')     
    print File.items()

    print 
    print File['Chombo_global'].attrs['SpaceDim']
    print
    print File['level_0'].items()
    #print
    #print File['level_0']['Processors'][:]
    print 
    print File['level_0']['boxes'][:]
    print 
    print File['level_0']['data:offsets=0'][:]
    print 
    print len(File['level_0']['data:datatype=0'][:])
    print 
    print File['level_0']['data_attributes'].attrs.items()
    print 
    print File['level_0']['data_attributes'].attrs.keys()
    
    ghost = File['level_0']['data_attributes'].attrs.get('ghost')
    print 'ghost=',ghost

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
    print total_box
    
    for i in range(dim*2):
        min_box_intvect[i]=min(total_box[:,i])
        max_box_intvect[i]=max(total_box[:,i])
    
    print 'lo=',min_box_intvect
    print 'hi=',max_box_intvect
    domain_box_intvect=min_box_intvect
    domain_box_intvect[dim:dim*2]=max_box_intvect[dim:dim*2]
    print 'domain=',domain_box_intvect
    
    shifter=np.zeros(dim*2)
    shifter=shifter.astype(int)
    for i in range(dim):
        shifter[i]=-domain_box_intvect[i]
    for i in range(dim):
        shifter[i+dim]=-domain_box_intvect[i]
    print 'domian_shifter=',shifter

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
    print 'num_cell_loc=',num_cell_loc
    prod_num_cell_loc = np.prod(num_cell_loc)
    print 'prod_num_cell_loc=',prod_num_cell_loc
   
    num_cell_loc_with_ghost=np.ones(dim)
    num_cell_loc_with_ghost=num_cell_loc_with_ghost.astype(int)
    for i in range(dim):
        num_cell_loc_with_ghost[i]=num_cell_loc[i]+2*ghost[i]
    print 'num_cell_loc_with_ghost=',num_cell_loc_with_ghost
    prod_num_cell_loc_with_ghost = np.prod(num_cell_loc_with_ghost)
    print 'prod_num_cell_loc_with_ghost=',prod_num_cell_loc_with_ghost
 
    num_cell_total=np.ones(dim)
    num_cell_total=num_cell_total.astype(int)
    for i in range(dim):
        xcell_beg=min_box_intvect[i]
        xcell_fin=max_box_intvect[i+dim]
        num_xcell=xcell_fin-xcell_beg+1
        #print xcell_beg,xcell_fin,num_xcell
        num_cell_total[i]=num_xcell
    print 'num_cell_total=',num_cell_total
    prod_num_cell_total = np.prod(num_cell_total)
    print 'prod_num_cell_total=',prod_num_cell_total

    decomposition=np.ones(dim)
    decomposition=decomposition.astype(int)
    for i in range(dim):
        decomposition[i] = num_cell_total[i]/num_cell_loc[i]
    print 'decomposition=',decomposition

    num_cell_total_with_ghost=np.ones(dim)
    num_cell_total_with_ghost=num_cell_total_with_ghost.astype(int)
    for i in range(dim):
        num_cell_total_with_ghost[i]=(num_cell_loc[i]+2*ghost[i])*decomposition[i]
    print 'num_cell_total_with_ghost=',num_cell_total_with_ghost
    prod_num_cell_total_with_ghost = np.prod(num_cell_total_with_ghost)
    print 'prod_num_cell_total_with_ghost=',prod_num_cell_total_with_ghost

    num_cell_total_with_outer_ghost=np.ones(dim)
    num_cell_total_with_outer_ghost=num_cell_total_with_outer_ghost.astype(int)
    for i in range(dim):
        num_cell_total_with_outer_ghost[i]=num_cell_loc[i]*decomposition[i]+2*ghost[i]
    print 'num_cell_total_with_outer_ghost=',num_cell_total_with_outer_ghost
    prod_num_cell_total_with_outer_ghost = np.prod(num_cell_total_with_outer_ghost)
    print 'prod_num_cell_total_with_outer_ghost=',prod_num_cell_total_with_outer_ghost


    #import potential data
    data = File['level_0']['data:datatype=0'][:]
    previous_index=0
    
    dataNd_potential_with_ghost=np.linspace(0.0,0.1,num=prod_num_cell_total_with_ghost).reshape((num_cell_total_with_ghost))
    dataNd_potential=np.linspace(0.0,0.1,num=prod_num_cell_total).reshape((num_cell_total))
    dataNd_potential_with_outer_ghost=np.linspace(0.0,0.1,num=prod_num_cell_total_with_outer_ghost).reshape((num_cell_total_with_outer_ghost))
    
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
        for j in range(dim):
            temp_decomp= cells_shift[j]/num_cell_loc[j]
            cells_shift_with_ghost[j]=cells_shift[j]+ghost[j]*temp_decomp*2
            cells_shift_with_ghost[j+dim]=(cells_shift[j]+ghost[j]*temp_decomp*2)+num_cell_loc_with_ghost[j]-1

        #print 'cells=',cells
        #print 'cells_shift=',cells_shift
        #print 'cells_shift_with_ghost=',cells_shift_with_ghost
            
        if dim<3:
            dataNd_loc_with_ghost=data[previous_index:prod_num_cell_loc_with_ghost*(i+1)].reshape((num_cell_loc_with_ghost[dir_x],num_cell_loc_with_ghost[dir_y]),order='F')
        else:
            dataNd_loc_with_ghost=data[previous_index:prod_num_cell_loc_with_ghost*(i+1)].reshape((num_cell_loc_with_ghost[dir_x],num_cell_loc_with_ghost[dir_y],num_cell_loc_with_ghost[dir_z]),order='F')
    
        previous_index=prod_num_cell_loc_with_ghost*(i+1)
    
        #print cells_shift
        #print cells_shift[dir_x],cells_shift[dim+dir_x]+1
        #print cells_shift[dir_y],cells_shift[dim+dir_y]+1
        #print cells_shift[dir_z],cells_shift[dim+dir_z]+1
    
        dataNd_potential_with_ghost[cells_shift_with_ghost[dir_x]:cells_shift_with_ghost[dim+dir_x]+1, cells_shift_with_ghost[dir_y]:cells_shift_with_ghost[dim+dir_y]+1, cells_shift_with_ghost[dir_z]:cells_shift_with_ghost[dim+dir_z]+1]=dataNd_loc_with_ghost
    
    
    File.close()


    print 'num_decomposition=',num_decomposition

    print 'dataNd_potential_with_ghost.shape=',dataNd_potential_with_ghost.shape
    print 'dataNd_potential_with_outer_ghost.shape=',dataNd_potential_with_outer_ghost.shape
    print 'dataNd_potential.shape=',dataNd_potential.shape
    print 'decomposition=',decomposition
    print 'ghost=',ghost

    print num_cell_total_with_ghost
    print num_cell_total_with_outer_ghost
    print num_cell_total
    print num_cell_loc

    #removing all ghost cells
    for i in range(num_cell_total[0]):
        current_decomp_i=i/num_cell_loc[0]
        for j in range(num_cell_total[1]):
            current_decomp_j=j/num_cell_loc[1]
            for k in range(num_cell_total[2]):
                current_decomp_k=k/num_cell_loc[2]
                dataNd_potential[i,j,k]=dataNd_potential_with_ghost[ghost[0]+ghost[0]*2*current_decomp_i+i,ghost[1]+ghost[1]*2*current_decomp_j+j,ghost[2]+ghost[2]*2*current_decomp_k+k]

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
                dataNd_potential_with_outer_ghost[i,j,k]=dataNd_potential_with_ghost[ghost[0]*2*(current_decomp_i-last_decomp_i-1)+i,ghost[1]*2*(current_decomp_j-last_decomp_j-1)+j,ghost[2]*2*(current_decomp_k-last_decomp_k-1)+k]

    print 'dataNd_potential_with_ghost.shape=',dataNd_potential_with_ghost.shape
    print 'dataNd_potential_with_outer_ghost.shape=',dataNd_potential_with_outer_ghost.shape
    print 'dataNd_potential.shape=',dataNd_potential.shape

    #potential without ghost
    wh=1 # 0: black background, 1: whithe background
    fig=mlab.figure(bgcolor=(wh,wh,wh),size=(600,600))
    mlab.contour3d(dataNd_potential,contours=10,transparent=True,opacity=0.8)
    ol=mlab.outline(color=(1-wh,1-wh,1-wh))
    ax = mlab.axes(nb_labels=5,ranges=[0,1,0,1,0,1])
    ax.axes.property.color=(1-wh,1-wh,1-wh)
    ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.label_format='%.2f'
    cb=mlab.colorbar(title='Potential', orientation='vertical')
    cb.title_text_property.color=(1-wh,1-wh,1-wh)
    cb.label_text_property.color=(1-wh,1-wh,1-wh)

    mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')

    mlab.savefig('fig10_mlab_iso.png')
    fig.scene.save_ps('fig10_mlab_iso.eps')
    #mlab.close(all=True)

    #potential with outer ghost 
    wh=1 # 0: black background, 1: whithe background
    fig=mlab.figure(bgcolor=(wh,wh,wh),size=(600,600))
    mlab.contour3d(dataNd_potential_with_outer_ghost,contours=10,transparent=True,opacity=0.8,extent=[0,num_cell_total_with_outer_ghost[0],0,num_cell_total_with_outer_ghost[1],0,num_cell_total_with_outer_ghost[2]])
    ol=mlab.outline(color=(1-wh,1-wh,1-wh))
    ax = mlab.axes(nb_labels=5,ranges=[0,1,0,1,0,1])
    ax.axes.property.color=(1-wh,1-wh,1-wh)
    ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.label_format='%.2f'
    cb=mlab.colorbar(title='Potential', orientation='vertical')
    cb.title_text_property.color=(1-wh,1-wh,1-wh)
    cb.label_text_property.color=(1-wh,1-wh,1-wh)
    mlab.plot3d([ghost[0]         +1, num_cell_total[0]+1], [ghost[1]         +1, ghost[1]         +1], [ghost[2]         +1, ghost[2]         +1], color=(1,0,0), tube_radius=0.1)
    mlab.plot3d([ghost[0]         +1, num_cell_total[0]+1], [ghost[1]         +1, ghost[1]         +1], [num_cell_total[2]+1, num_cell_total[2]+1], color=(1,0,0), tube_radius=0.1)
    mlab.plot3d([ghost[0]         +1, num_cell_total[0]+1], [num_cell_total[1]+1, num_cell_total[1]+1], [ghost[2]         +1,ghost[2]          +1], color=(1,0,0), tube_radius=0.1)
    mlab.plot3d([ghost[0]         +1, num_cell_total[0]+1], [num_cell_total[1]+1, num_cell_total[1]+1], [num_cell_total[2]+1, num_cell_total[2]+1], color=(1,0,0), tube_radius=0.1)

    mlab.plot3d([ghost[0]         +1, ghost[0]         +1], [ghost[1]         +1, num_cell_total[1]+1], [ghost[2]         +1, ghost[2]         +1], color=(0,1,0), tube_radius=0.1)
    mlab.plot3d([ghost[0]         +1, ghost[0]         +1], [ghost[1]         +1, num_cell_total[1]+1], [num_cell_total[2]+1, num_cell_total[2]+1], color=(0,1,0), tube_radius=0.1)
    mlab.plot3d([num_cell_total[0]+1, num_cell_total[0]+1], [ghost[1]         +1, num_cell_total[1]+1], [ghost[2]         +1,ghost[2]          +1], color=(0,1,0), tube_radius=0.1)
    mlab.plot3d([num_cell_total[0]+1, num_cell_total[0]+1], [ghost[1]         +1, num_cell_total[1]+1], [num_cell_total[2]+1, num_cell_total[2]+1], color=(0,1,0), tube_radius=0.1)

    mlab.plot3d([ghost[0]         +1, ghost[0]         +1], [ghost[1]         +1, ghost[1]         +1],[ghost[2]          +1, num_cell_total[2]+1],  color=(0,0,1), tube_radius=0.1)
    mlab.plot3d([ghost[0]         +1, ghost[0]         +1], [num_cell_total[1]+1, num_cell_total[1]+1],[ghost[2]          +1, num_cell_total[2]+1],  color=(0,0,1), tube_radius=0.1)
    mlab.plot3d([num_cell_total[0]+1, num_cell_total[0]+1], [ghost[1]         +1, ghost[1]         +1],[ghost[2]          +1, num_cell_total[2]+1],  color=(0,0,1), tube_radius=0.1)
    mlab.plot3d([num_cell_total[0]+1, num_cell_total[0]+1], [num_cell_total[1]+1, num_cell_total[1]+1],[ghost[2]          +1, num_cell_total[2]+1],  color=(0,0,1), tube_radius=0.1)

    mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')
    mlab.show()

    mlab.savefig('fig10_mlab_iso_outer_ghost.png')
    fig.scene.save_ps('fig10_mlab_iso_outer_ghost.eps')
    #mlab.close(all=True)
       
    #potential 3D mlab slice without ghost
    wh=1 # 0: black background, 1: whithe background
    fig=mlab.figure(bgcolor=(wh,wh,wh),size=(600,600))
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(dataNd_potential),plane_orientation='x_axes',slice_index=x_pt)
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(dataNd_potential),plane_orientation='y_axes',slice_index=y_pt)
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(dataNd_potential),plane_orientation='z_axes',slice_index=z_pt)
    ol=mlab.outline(color=(1-wh,1-wh,1-wh))
    ax = mlab.axes(nb_labels=5,ranges=[0,1,0,1,0,1])
    ax.axes.property.color=(1-wh,1-wh,1-wh)
    ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.label_format='%.2f'
    cb=mlab.colorbar(title='Density', orientation='vertical')
    cb.title_text_property.color=(1-wh,1-wh,1-wh)
    cb.label_text_property.color=(1-wh,1-wh,1-wh)
    mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')
    mlab.savefig('fig10_mlab_slice.png')
    fig.scene.save_ps('fig10_mlab_slice.eps')
    mlab.show() 

    


















#############################################################
File =h5py.File(hdffilename,'r')     

print File.items()
#print 
#print File['Chombo_global'].attrs['SpaceDim']
#print
#print File['level_0'].items()
#print
#print File['level_0']['Processors'][:]
#print 
#print File['level_0']['boxes'][:]
#print 
#print File['level_0']['data:offsets=0'][:]
#print 
#print len(File['level_0']['data:datatype=0'][:])
#print 
#print File['level_0']['data_attributes'].attrs.items()

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

print 'lo=',min_box_intvect
print 'hi=',max_box_intvect
domain_box_intvect=min_box_intvect
domain_box_intvect[dim:dim*2]=max_box_intvect[dim:dim*2]
print 'domain=',domain_box_intvect

shifter=np.zeros(dim*2)
shifter=shifter.astype(int)
for i in range(dim):
    shifter[i]=-domain_box_intvect[i]
for i in range(dim):
    shifter[i+dim]=-domain_box_intvect[i]
print 'domian_shifter=',shifter

if dim<5:
    dir_x=0
    dir_y=1
    dir_vpar=2
    dir_mu=3
else:
    dir_x=0
    dir_y=1
    dir_z=2
    dir_vpar=3
    dir_mu=4

if dim>3:
    cfg_dim=dim-2
else:
    cfg_dim=dim


num_cell_total=np.ones(dim)
num_cell_total=num_cell_total.astype(int)
for i in range(dim):
    xcell_beg=min_box_intvect[i]
    xcell_fin=max_box_intvect[i+dim]
    num_xcell=xcell_fin-xcell_beg+1
    #print xcell_beg,xcell_fin,num_xcell
    num_cell_total[i]=num_xcell
print 'num_cell_total=',num_cell_total
prod_num_cell_total = np.prod(num_cell_total)
print 'prod_num_cell_total=',prod_num_cell_total

num_cell_loc=np.ones(dim)
num_cell_loc=num_cell_loc.astype(int)
for i in range(dim):
    xcell_beg=boxes[0][i]
    xcell_fin=boxes[0][i+dim]
    num_xcell=xcell_fin-xcell_beg+1
    #print xcell_beg,xcell_fin,num_xcell
    num_cell_loc[i]=num_xcell
print 'num_cell_loc=',num_cell_loc
prod_num_cell_loc = np.prod(num_cell_loc)
print 'prod_num_cell_loc=',prod_num_cell_loc
     
data = File['level_0']['data:datatype=0'][:]
previous_index=0

if dim<5:
    dataNd=np.linspace(0.0,0.1,num=prod_num_cell_total).reshape((num_cell_total[0],num_cell_total[1],num_cell_total[2],num_cell_total[3]) )
else:
    dataNd=np.linspace(0.0,0.1,num=prod_num_cell_total).reshape((num_cell_total))
num_cell_loc=num_cell_loc.astype(int)

cells_shift=np.zeros(dim*2)
cells_shift=cells_shift.astype(int)

for i in range(num_decomposition):
    cells = File['level_0']['boxes'][i]
    sys.stdout.write('.')
    sys.stdout.flush()
    #print 'cells=',cells
    
    for j in range(len(cells_shift)):
        cells_shift[j]=cells[j]+shifter[j]
        
    if dim<5:
        dataNd_loc=data[previous_index:prod_num_cell_loc*(i+1)].reshape((num_cell_loc[dir_x],num_cell_loc[dir_y],num_cell_loc[dir_vpar],num_cell_loc[dir_mu]),order='F')
    else:
        #print previous_index
        #print prod_num_cell_loc*(i+1)
        #print prod_num_cell_loc
        #print prod_num_cell_total
        #print (prod_num_cell_loc*(i+1)-previous_index)
        #print prod_num_cell_total/(prod_num_cell_loc*(i+1)-previous_index)

        dataNd_loc=data[previous_index:prod_num_cell_loc*(i+1)].reshape((num_cell_loc[dir_x],num_cell_loc[dir_y],num_cell_loc[dir_z],num_cell_loc[dir_vpar],num_cell_loc[dir_mu]),order='F')

    previous_index=prod_num_cell_loc*(i+1)

    #print cells_shift
    #print cells_shift[dir_x],cells_shift[dim+dir_x]+1
    #print cells_shift[dir_y],cells_shift[dim+dir_y]+1
    #print cells_shift[dir_z],cells_shift[dim+dir_z]+1
    #print cells_shift[dir_vpar],cells_shift[dim+dir_vpar]+1
    #print cells_shift[dir_mu],cells_shift[dim+dir_mu]+1



    dataNd[cells_shift[dir_x]:cells_shift[dim+dir_x]+1, cells_shift[dir_y]:cells_shift[dim+dir_y]+1, cells_shift[dir_z]:cells_shift[dim+dir_z]+1, cells_shift[dir_vpar]:cells_shift[dim+dir_vpar]+1, cells_shift[dir_mu]:cells_shift[dim+dir_mu]+1]=dataNd_loc


File.close()
print 'num_decomposition=',num_decomposition
print 'dataNd.shape=',dataNd.shape

num_xcell=num_cell_total[dir_x] #
num_ycell=num_cell_total[dir_y] #
if dim>=5:
    num_zcell=num_cell_total[dir_z] #
num_vparcell=num_cell_total[dir_vpar] #
num_mucell=num_cell_total[dir_mu] #

#phase
VPAR,MU = np.mgrid[-1:1:(num_vparcell*1j),0:1:(num_mucell*1j)]

##refine #deprecated - causes producing to big memory
#if dim<5:
#    rbf = interpolate.Rbf(VPAR.ravel(), MU.ravel(), dataNd[x_pt,y_pt,:,:].ravel(), smooth=rbf_smooth)
#else:
#    rbf = interpolate.Rbf(VPAR.ravel(), MU.ravel(), dataNd[x_pt,y_pt,z_pt,:,:].ravel(), smooth=rbf_smooth)
#    
#VPAR_REFINE, MU_REFINE = np.mgrid[-1:1:(num_vparcell*1j*vpar_rbf_refine), 0:1:(num_mucell*1j*mu_rbf_refine)]
#saved_rbf=rbf(VPAR_REFINE,MU_REFINE)


#### READ INPUT DECK
fname=find('*.in', './')

with open(fname[0], 'r') as f:
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
            if 'units.number_density' in lhsrhs[0]:
                units_number_density =float(lhsrhs[1])
                print 'IN:units_number_density = ',units_number_density
            if 'units.temperature' in lhsrhs[0]:
                units_temperature = float(lhsrhs[1])
                print 'IN:units_temperature = ',units_temperature
            if 'units.length' in lhsrhs[0]:
                units_length = float(lhsrhs[1])
                print 'IN:units_length= ',units_length
            if 'units.mass' in lhsrhs[0]:
                units_mass = float(lhsrhs[1])
                print 'IN:units_mass = ',units_mass
            if 'units.magnetic_field' in lhsrhs[0]:
                units_magnetic_field = float(lhsrhs[1])
                print 'IN:units_magnetic_field = ',units_magnetic_field
            if 'boltzmann_electron.temperature' in lhsrhs[0]:
                boltzmann_electron_temperature = float(lhsrhs[1])
                print 'IN:boltzmann_electron_temperature = ',boltzmann_electron_temperature
            if 'gksystem.magnetic_geometry_mapping.slab.Bz_inner' in lhsrhs[0]:
                bz_inner = float(lhsrhs[1])
                print 'IN:bz_inner = ',bz_inner
            if 'gksystem.magnetic_geometry_mapping.slab.By_inner' in lhsrhs[0]:
                by_inner = float(lhsrhs[1])
                print 'IN:by_inner = ',by_inner
            if 'gksystem.magnetic_geometry_mapping.slab.x_max' in lhsrhs[0]:
                x_max = float(lhsrhs[1])
                print 'IN:x_max = ',x_max
            if 'gksystem.magnetic_geometry_mapping.slab.y_max' in lhsrhs[0]:
                y_max = float(lhsrhs[1])
                print 'IN:y_max = ',y_max
            if 'gksystem.magnetic_geometry_mapping.slab.z_max' in lhsrhs[0]:
                z_max = float(lhsrhs[1])
                print 'IN:z_max = ',z_max
            if 'kinetic_species.1.mass' in lhsrhs[0]:
                ion_mass = float(lhsrhs[1])
                print 'IN:ion_mass = ',ion_mass
            if 'kinetic_species.2.mass' in lhsrhs[0]:
                elec_mass = float(lhsrhs[1])
                print 'IN:elec_mass = ',elec_mass

            if 'phase_space_mapping.v_parallel_max' in lhsrhs[0]:
                v_parallel_max= float(lhsrhs[1])
                print 'IN:v_parallel_max= ',v_parallel_max
            if 'phase_space_mapping.mu_max' in lhsrhs[0]:
                mu_max= float(lhsrhs[1])
                print 'IN:mu_max= ',mu_max


            if '.N0_grid_func.function' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                n0_grid_func=lhsrhs[1][1:-1] #remove double quotes
                n0_grid_func=n0_grid_func.lstrip()
                n0_grid_func=n0_grid_func.rstrip()
                n0_grid_func=n0_grid_func.replace('^','**')
                print 'IN:n0_grid_func = ',n0_grid_func
                m_y=0.0;
                m_x=0.0;
                m_z=0.0;
                #parse background 
                tokens=findpertbation(n0_grid_func,'exp(')
                #print 'exp():',tokens
                for token in tokens:
                    #print "For ",token
                    #temp_y=findmodenumber(token,'y')
                    temp_x=findmodenumber(token,'x')
                    #temp_z=findmodenumber(token,'z')
                    #if abs(temp_y)>abs(m_y):
                    #        m_y=temp_y
                    if abs(temp_x)>abs(m_x):
                            m_x=temp_x
                    #if abs(temp_z)>abs(m_z):
                    #        m_z=temp_z
                deltaL_analytic = -1.0/m_x/2.0/np.pi*x_max*100 #in cm
                print 'IN:deltaL_analytic=',deltaL_analytic

                #parse perturbations
                m_y=0.0;
                m_x=0.0;
                m_z=0.0;

                tokens=findpertbation(n0_grid_func,'sin(')
                #print 'sin():',tokens
                for token in tokens:
                    #print "For ",token
                    temp_y=findmodenumber(token,'y')
                    temp_x=findmodenumber(token,'x')
                    temp_z=findmodenumber(token,'z')
                    if abs(temp_y)>abs(m_y):
                            m_y=temp_y
                    if abs(temp_x)>abs(m_x):
                            m_x=temp_x
                    if abs(temp_z)>abs(m_z):
                            m_z=temp_z

                print 'IN:m_y=',m_y
                print 'IN:m_x=',m_x
                print 'IN:m_z=',m_z

                tokens=findpertbation(n0_grid_func,'cos(')
                #print 'cos():',tokens
                for token in tokens:
                    #print "For ",token
                    temp_y=findmodenumber(token,'y')
                    temp_x=findmodenumber(token,'x')
                    temp_z=findmodenumber(token,'z')
                    if abs(temp_y)>abs(m_y):
                            m_y=temp_y
                    if abs(temp_x)>abs(m_x):
                            m_x=temp_x
                    if abs(temp_z)>abs(m_z):
                            m_z=temp_z

                print 'IN:m_y=',m_y
                print 'IN:m_x=',m_x
                print 'IN:m_z=',m_z
                
            if '.T0_grid_func.constant' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                t0_grid_func=float(lhsrhs[1])
                print 'IN:t0_grid_func = ',t0_grid_func
            if '.T0_grid_func.value' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                t0_grid_func=float(lhsrhs[1])
                print 'IN:t0_grid_func = ',t0_grid_func
            if '.eT0_grid_func.constant' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                et0_grid_func=float(lhsrhs[1])
                print 'IN:et0_grid_func = ',et0_grid_func
            if '.eT0_grid_func.value' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                et0_grid_func=float(lhsrhs[1])
                print 'IN:et0_grid_func = ',et0_grid_func
            
            if 'gksystem.num_cells' in lhsrhs[0]:
                num_cells=lhsrhs[1].split()
                #print 'IN:num_cells = ',num_cells
                if len(num_cells) == 4:
                    x_cells=int(num_cells[0])
                    #print 'IN:x_cells = ', x_cells
                    y_cells=int(num_cells[1])
                    #print 'IN:y_cells = ', y_cells
                elif len(num_cells) == 5:
                    x_cells=int(num_cells[0])
                    #print 'IN:x_cells = ', x_cells
                    y_cells=int(num_cells[1])
                    #print 'IN:y_cells = ', y_cells
                    z_cells=int(num_cells[2])
                    #print 'IN:z_cells = ', z_cells
            if '.history_indices' in lhsrhs[0]:
                history_indices=lhsrhs[1].split()
                if len(history_indices) == 2:
                    x_index=int(history_indices[0])
                    #print 'IN:x_index = ', x_index
                    y_index=int(history_indices[1])
                    #print 'IN:y_index = ', y_index
                elif len(history_indices) == 3:
                    x_index=int(history_indices[0])
                    #print 'IN:x_index = ', x_index
                    y_index=int(history_indices[1])
                    #print 'IN:y_index = ', y_index
                    z_index=int(history_indices[2])
                    #print 'IN:z_index = ', z_index



#reconstruct from model Maxwellian
Bhat = np.sqrt(bz_inner**2+by_inner**2)

if 'electron' in hdffilename:
    mhat = elec_mass
    That = et0_grid_func
    print 'electron (mass, t0) = ', mhat, That
elif 'hydrogen' in hdffilename:
    mhat = ion_mass
    That = t0_grid_func
    print 'hydrogen (mass, t0) = ', mhat, That
else:
    mhat = 1.0
    That = 1.0
    print 'default (mass, t0) = ', mhat, That


transformations = (standard_transformations + (implicit_multiplication_application,))
pe=parse_expr(n0_grid_func,transformations=transformations)
f_n0_grid_func= lambdify((x,y,z),pe)

in_f_n0 = f_n0_grid_func(0,0,0)
if in_f_n0 >0.0:
    nhat = in_f_n0
    print 'nhat = ', nhat
else:
    nhat =1.0
    print 'nhat(default) = ', nhat

Vpar_max = v_parallel_max
Mu_max = mu_max
print '(vpar_max, mu_max) = ', Vpar_max, Mu_max


vpar_cell_dim_begin = -1.0*Vpar_max+(Vpar_max*1.0+Vpar_max*1.0)/num_vparcell/2.0
vpar_cell_dim_end   =  1.0*Vpar_max-(Vpar_max*1.0+Vpar_max*1.0)/num_vparcell/2.0
mu_cell_dim_begin = Mu_max*0.0+(Mu_max*1.0-Mu_max*0.0)/num_mucell/2.0
mu_cell_dim_end   = Mu_max*1.0-(Mu_max*1.0-Mu_max*0.0)/num_mucell/2.0
VPAR_CELL,MU_CELL = np.mgrid[vpar_cell_dim_begin:vpar_cell_dim_end:(num_vparcell*1j),mu_cell_dim_begin:mu_cell_dim_end:(num_mucell*1j)]

#VPAR_SCALE = VPAR_CELL[:,0]*np.sqrt(mhat) #for trunk
VPAR_SCALE = VPAR_CELL[:,0] #for mass dependent normalization
MU_SCALE = MU_CELL[0,:]

coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)

#f_model = np.zeros((num_vparcell,num_mucell))
#for i in range(num_vparcell):
#    for j in range(num_mucell):
#        f_model[i][j]=coef_maxwell*np.exp(-0.5*(VPAR_SCALE[i]**2+MU_SCALE[j]*Bhat)/That)
#
#if dim<5:
#    delta_f_model = (f_model-dataNd[x_pt,y_pt,:,:])
#else:
#    delta_f_model = (f_model-dataNd[x_pt,y_pt,z_pt,:,:])

#least square fitting on a slice of MU index=0
guess_den =coef_maxwell 
guess_temp =1.0/(2.0*That)
guess_shift =0.0
if dim<5:
    optimize_func = lambda z: z[0]*np.exp(-z[1]*(VPAR_SCALE-z[2])**2)-dataNd[x_pt,y_pt,:,0] 
else:
    optimize_func = lambda z: z[0]*np.exp(-z[1]*(VPAR_SCALE-z[2])**2)-dataNd[x_pt,y_pt,z_pt,:,0] 

est_den, est_temp, est_shift = leastsq(optimize_func, [guess_den, guess_temp, guess_shift])[0]
fitted_f = est_den*np.exp(-est_temp*VPAR_SCALE**2)
t_fit = 1.0/(est_temp*2.0)
n_fit = est_den*np.sqrt(np.pi)/(0.5*mhat/t_fit)**(1.5)/np.exp(-MU_SCALE[0]*Bhat/2.0/t_fit)
vshift_fit = est_shift
print 't_fit = ',t_fit
print 'n_fit = ',n_fit
print 'vshift_fit = ',est_shift

#density sum
if dim<5:
    X,Y = np.mgrid[0:1:(num_xcell*1j),0:1:(num_ycell*1j)]
    f_vpar_mu_sum = np.zeros((num_xcell,num_ycell))
    delta_Mu = 1.0*Mu_max/num_mucell
    delta_Vpar = 2.0*Vpar_max/num_vparcell
    for i in range(num_xcell):
        for j in range(num_ycell):
            sumovervparandmu=0.0
            for k in range(num_vparcell):
                sumovermu=0.0
                for l in range(num_mucell):
                    sumovermu=sumovermu+dataNd[i,j,k,l]
                sumovermu=sumovermu*delta_Mu
                sumovervparandmu=sumovervparandmu+sumovermu
            sumovervparandmu=sumovervparandmu*delta_Vpar 
            f_vpar_mu_sum[i,j]=f_vpar_mu_sum[i,j]+sumovervparandmu
else:
    X,Y,Z = np.mgrid[0:1:(num_xcell*1j),0:1:(num_ycell*1j),0:1:(num_zcell*1j)]
    f_vpar_mu_sum = np.zeros((num_xcell,num_ycell,num_zcell))
    delta_Mu = 1.0*Mu_max/num_mucell
    delta_Vpar = 2.0*Vpar_max/num_vparcell
    for i in range(num_xcell):
        for j in range(num_ycell):
            for k in range(num_zcell):
                sumovervparandmu=0.0
                for l in range(num_vparcell):
                    sumovermu=0.0
                    for m in range(num_mucell):
                        sumovermu=sumovermu+dataNd[i,j,k,l,m]
                    sumovermu=sumovermu*delta_Mu
                    sumovervparandmu=sumovervparandmu+sumovermu
                sumovervparandmu=sumovervparandmu*delta_Vpar 
                f_vpar_mu_sum[i,j,k]=f_vpar_mu_sum[i,j,k]+sumovervparandmu
print f_vpar_mu_sum.shape


#sum over mu
f_mu_sum = np.zeros(num_vparcell)
if dim<5:
    for i in range(num_vparcell):
        for j in range(num_mucell):
            f_mu_sum[i]=f_mu_sum[i]+dataNd[x_pt,y_pt,i,j]
        f_mu_sum[i]=f_mu_sum[i]/num_mucell
else:
    for i in range(num_vparcell):
        for j in range(num_mucell):
            f_mu_sum[i]=f_mu_sum[i]+dataNd[x_pt,y_pt,z_pt,i,j]
        f_mu_sum[i]=f_mu_sum[i]/num_mucell

#f_mu_sum_rbf = np.zeros(len(VPAR_REFINE[:,0]))
#for i in range(len(VPAR_REFINE[:,0])):
#    for j in range(len(VPAR_REFINE[0,:])):
#        f_mu_sum_rbf[i]=f_mu_sum_rbf[i]+saved_rbf[i,j]
#    f_mu_sum_rbf[i]=f_mu_sum_rbf[i]/len(VPAR_REFINE[0,:])



##raw density 2D 
#init_plotting()
#plt.subplot(111)
##plt.gca().margins(0.1, 0.1)
#if dim<5:
#    im=plt.contourf(X,Y,f_vpar_mu_sum,100)
#else:
#    im=plt.contourf(X[:,:,0],Y[:,:,0],f_vpar_mu_sum[:,:,z_pt],100)
#plt.title(u'n(X,Y)')
#plt.xlabel(u'X')
#plt.ylabel(u'Y')
#plt.colorbar()
#plt.tight_layout()
#plt.savefig('fig1.png')
#plt.savefig('fig1.eps')
#plt.close('all')

#raw density 2D imshow
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    im=plt.imshow(f_vpar_mu_sum.T,interpolation='none',origin="lower",extent=[0,1,0,1],aspect=1.0)
else:
    im=plt.imshow(f_vpar_mu_sum[:,:,z_pt].T,interpolation='none',origin="lower",extent=[0,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
plt.title(u'n(X,Y)')
plt.xlabel(u'X')
plt.ylabel(u'Y')
add_colorbar(im)
plt.tight_layout()
plt.savefig('fig1_im.png')
plt.savefig('fig1_im.eps')
plt.close('all')

#raw density 2D imshow
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    im=plt.imshow(f_vpar_mu_sum.T,interpolation='spline16',origin="lower",extent=[0,1,0,1],aspect=1.0)
else:
    im=plt.imshow(f_vpar_mu_sum[:,:,z_pt].T,interpolation='spline16',origin="lower",extent=[0,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
plt.title(u'n(X,Y)')
plt.xlabel(u'X')
plt.ylabel(u'Y')
add_colorbar(im)
plt.tight_layout()
plt.savefig('fig1_im_interp.png')
plt.savefig('fig1_im_interp.eps')
plt.close('all')



if dim>=5:
    #raw density 3D mlab
    wh=1 # 0: black background, 1: whithe background
    fig=mlab.figure(bgcolor=(wh,wh,wh),size=(600,600))
    mlab.contour3d(f_vpar_mu_sum,contours=10,transparent=True,opacity=0.8)
    ol=mlab.outline(color=(1-wh,1-wh,1-wh))
    ax = mlab.axes(nb_labels=5,ranges=[0,1,0,1,0,1])
    ax.axes.property.color=(1-wh,1-wh,1-wh)
    ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.label_format='%.2f'
    cb=mlab.colorbar(title='Density', orientation='vertical')
    cb.title_text_property.color=(1-wh,1-wh,1-wh)
    cb.label_text_property.color=(1-wh,1-wh,1-wh)

    mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')

    mlab.savefig('fig1_mlab_iso.png')
    fig.scene.save_ps('fig1_mlab_iso.eps')
    #arr=mlab.screenshot()
    #plt.imshow(arr)
    #plt.axis('off')
    #plt.savefig('fig1_mlab_iso.eps')
    #plt.savefig('fig1_mlab_iso.pdf')
    #fig.scene.close()
    mlab.close(all=True)
    
    
    #raw density 3D mlab volume
    wh=0 # 0: black background, 1: whithe background
    fig=mlab.figure(bgcolor=(wh,wh,wh),size=(600,600))
    mlab.pipeline.volume(mlab.pipeline.scalar_field(f_vpar_mu_sum))
    ol=mlab.outline(color=(1-wh,1-wh,1-wh))
    ax = mlab.axes(nb_labels=5,ranges=[0,1,0,1,0,1])
    ax.axes.property.color=(1-wh,1-wh,1-wh)
    ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.label_format='%.2f'
    cb=mlab.colorbar(title='Density', orientation='vertical')
    cb.title_text_property.color=(1-wh,1-wh,1-wh)
    cb.label_text_property.color=(1-wh,1-wh,1-wh)
    mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')
    mlab.savefig('fig1_mlab_volume.png')
    fig.scene.save_ps('fig1_mlab_volume.eps')
    #fig.scene.close()
    mlab.close(all=True)
    
    #raw density 3D mlab slace
    wh=1 # 0: black background, 1: whithe background
    fig=mlab.figure(bgcolor=(wh,wh,wh),size=(600,600))
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(f_vpar_mu_sum),plane_orientation='x_axes',slice_index=x_pt)
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(f_vpar_mu_sum),plane_orientation='y_axes',slice_index=y_pt)
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(f_vpar_mu_sum),plane_orientation='z_axes',slice_index=z_pt)
    ol=mlab.outline(color=(1-wh,1-wh,1-wh))
    ax = mlab.axes(nb_labels=5,ranges=[0,1,0,1,0,1])
    ax.axes.property.color=(1-wh,1-wh,1-wh)
    ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
    ax.axes.label_format='%.2f'
    cb=mlab.colorbar(title='Density', orientation='vertical')
    cb.title_text_property.color=(1-wh,1-wh,1-wh)
    cb.label_text_property.color=(1-wh,1-wh,1-wh)
    mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')
    mlab.savefig('fig1_mlab_slice.png')
    fig.scene.save_ps('fig1_mlab_slice.eps')
    #fig.scene.close()
    mlab.close(all=True)

#diagnostic point
if dim<5:
    init_plotting()
    fig=plt.subplot(111)
    plt.contourf(X,Y,f_vpar_mu_sum)
    plt.colorbar();
    plt.scatter(float(x_pt)/num_xcell,float(y_pt)/num_ycell)
    plt.gca().margins(0.0, 0.0)
    plt.title(u'n(X,Y)')
    plt.xlabel(u'X')
    plt.ylabel(u'Y')
    plt.tight_layout()
    plt.savefig('fig1_diag.png')
    plt.savefig('fig1_diag.eps')
    plt.close('all')


##raw f
#init_plotting()
#plt.subplot(111)
##plt.gca().margins(0.1, 0.1)
#if dim<5:
#    plt.contourf(VPAR,MU,dataNd[x_pt,y_pt,:,:],100)
#else:
#    plt.contourf(VPAR,MU,dataNd[x_pt,y_pt,z_pt,:,:],100)
#plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
#plt.xlabel(r'$\bar{v}_\parallel$')
#plt.ylabel(r'$\bar{\mu}$')
#plt.colorbar();
#plt.tight_layout()
#plt.savefig('fig2.png')
#plt.savefig('fig2.eps')
#plt.close('all')

#raw f imshow
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    im=plt.imshow(dataNd[x_pt,y_pt,:,:].T,interpolation='none',origin="lower",extent=[-1,1,0,1],aspect=1.0)
else:
    im=plt.imshow(dataNd[x_pt,y_pt,z_pt,:,:].T,interpolation='none',origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
add_colorbar(im)
plt.tight_layout()
plt.savefig('fig2_im.png')
plt.savefig('fig2_im.eps')
plt.close('all')

#raw f imshow
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    im=plt.imshow(dataNd[x_pt,y_pt,:,:].T,interpolation='spline16',origin="lower",extent=[-1,1,0,1],aspect=1.0)
else:
    im=plt.imshow(dataNd[x_pt,y_pt,z_pt,:,:].T,interpolation='spline16',origin="lower",extent=[-1,1,0,1],aspect=1.0)#float(num_ycell)/float(num_xcell))
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
add_colorbar(im)
plt.tight_layout()
plt.savefig('fig2_im_interp.png')
plt.savefig('fig2_im_interp.eps')
plt.close('all')

###reconstructed model maxwellian
#init_plotting()
#fig=plt.subplot(111)
##plt.gca().margins(0.1, 0.1)
##plt.contourf(VPAR_CELL, MU_CELL, f_model,100)
#plt.contourf(VPAR, MU, f_model,100)
#plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
#plt.xlabel(r'$\bar{v}_\parallel$')
#plt.ylabel(r'$\bar{\mu}$')
#plt.tight_layout()
#plt.colorbar();
#plt.savefig('fig2_model.png')
#plt.savefig('fig2_model.eps')
#plt.close('all')

###model maxwellian delta
#init_plotting()
#fig=plt.subplot(111)
##plt.gca().margins(0.1, 0.1)
#plt.contourf(VPAR, MU, delta_f_model,100)
#plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
#plt.xlabel(r'$\bar{v}_\parallel$')
#plt.ylabel(r'$\bar{\mu}$')
#plt.tight_layout()
#plt.colorbar();
#plt.savefig('fig2_delta_model.png')
#plt.savefig('fig2_delta_model.eps')
#plt.close('all')


#summed maxwellian 
init_plotting()
fig=plt.subplot(111)
plt.plot(VPAR[:,0], f_mu_sum )
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel('<f>')
plt.tight_layout()
plt.savefig('fig3.png')
plt.savefig('fig3.eps')
plt.close('all')

##summed maxwellian on refined 
#init_plotting()
#fig=plt.subplot(111)
#plt.plot(VPAR_REFINE[:,0], f_mu_sum_rbf )
#plt.xlabel(r'$\bar{v}_\parallel$')
#plt.ylabel('<f>')
#plt.tight_layout()
#plt.savefig('fig3_rbf.png')
#plt.savefig('fig3_rbf.eps')
#plt.close('all')

##compare mu slice with model
#init_plotting()
#fig=plt.subplot(111)
##plt.gca().margins(0.1, 0.1)
#if dim<5:
#    plt.plot(VPAR[:,0], f_model[:,0]/dataNd[x_pt,y_pt,:,0])
#else:
#    plt.plot(VPAR[:,0], f_model[:,0]/dataNd[x_pt,y_pt,z_pt,:,0])
#plt.xlabel(r'$\bar{v}_\parallel$')
#plt.ylabel(u'f_model/f(MU=0)')
#plt.tight_layout()
#plt.savefig('fig4_mu0_comp_model.png')
#plt.savefig('fig4_mu0_comp_model.eps')
#plt.close('all')

#compare mu slice with fitting
legend_maxwellian = 'MAXWELLIAN\n'+r'$n, T, V_s$'+' = (%.1f, %.1f, %.1g)'%(n_fit,t_fit,vshift_fit)
init_plotting()
fig=plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    plt.plot(VPAR[:,0], dataNd[x_pt,y_pt,:,0],linewidth=1.5,color='b',label='COGENT')
else:
    plt.plot(VPAR[:,0], dataNd[x_pt,y_pt,z_pt,:,0],linewidth=1.5,color='b',label='COGENT')
plt.plot(VPAR[:,0], fitted_f,linewidth=1.5,linestyle='--',color='k',label=legend_maxwellian)
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(u'f_(MU=0)')
#plt.gca().legend(bbox_to_anchor = (0.0, 0.2))
plt.gca().legend(loc='best')
plt.tight_layout()
plt.savefig('fig4_mu0_comp_fit.png')
plt.savefig('fig4_mu0_comp_fit.eps')
plt.close('all')


#collect figures
import Image
init_plotting('2x3')
f = pl.figure()
for n, fname in enumerate(('fig1_mlab_iso.png','fig1_mlab_slice.png','fig2_im.png','fig2_im_interp.png','fig3.png','fig4_mu0_comp_fit.png')):
     image=Image.open(fname)#.convert("L")
     arr=np.asarray(image)
     ax=f.add_subplot(2, 3, n+1)
     ax.axis('off')
     pl.imshow(arr)
pl.tight_layout()
pl.savefig('fig0.png')
pl.show()





