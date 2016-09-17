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


from plot_cogent_pack import find
from plot_cogent_pack import findpertbation
from plot_cogent_pack import findmodenumber


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
                if m_x==0:
                    m_x=1.0E50
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
print 'Bhat = ', Bhat

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



