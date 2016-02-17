import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import os, fnmatch
import ConfigParser

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result


#read input file
fname=find('*.in', './')

print '************ INPUT FILE *****************'
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
            #print type( lhsrhs[0])
            if 'units.number_density' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_number_density =float(lhsrhs[1])
                print 'IN:units_number_density = ',units_number_density
            if 'units.temperature' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_temperature = float(lhsrhs[1])
                print 'IN:units_temperature = ',units_temperature
            if 'units.length' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_length = float(lhsrhs[1])
                print 'IN:units_length= ',units_length
            if 'units.mass' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_mass = float(lhsrhs[1])
                print 'IN:units_mass = ',units_mass
            if 'units.magnetic_field' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_magnetic_field = float(lhsrhs[1])
                print 'IN:units_magnetic_field = ',units_magnetic_field
            if 'boltzmann_electron.temperature' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                boltzmann_electron_temperature = float(lhsrhs[1])
                print 'IN:boltzmann_electron_temperature = ',boltzmann_electron_temperature
            if 'gksystem.magnetic_geometry_mapping.slab.Bz_inner' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                bz_inner = float(lhsrhs[1])
                print 'IN:bz_inner = ',bz_inner
            if 'kinetic_species.1.mass' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                ion_mass = float(lhsrhs[1])
                print 'IN:ion_mass = ',ion_mass
            if '.N0_grid_func.function' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                n0_grid_func=lhsrhs[1][1:-1]
                n0_grid_func=n0_grid_func.replace('^','**')
                print 'IN:n0_grid_func = ',n0_grid_func
                if '*y' in n0_grid_func:
                    m_y = float(n0_grid_func[n0_grid_func.find('*y')-1 ]   )
                    print 'IN:m_y = ',m_y
            if 'gksystem.magnetic_geometry_mapping.slab.x_max' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                x_max = float(lhsrhs[1])
                print 'IN:x_max = ',x_max
            if 'gksystem.magnetic_geometry_mapping.slab.y_max' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                y_max = float(lhsrhs[1])
                print 'IN:y_max = ',y_max

            if '.T0_grid_func.constant' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                t0_grid_func=float(lhsrhs[1])
                print 'IN:t0_grid_func = ',t0_grid_func
            if '.T0_grid_func.value' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                t0_grid_func=float(lhsrhs[1])
                print 'IN:t0_grid_func = ',t0_grid_func
            if 'gksystem.num_cells' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                num_cells=lhsrhs[1].split()
                #print 'IN:num_cells = ',num_cells
                if len(num_cells) == 4:
                    x_cells=int(num_cells[0])
                    print 'IN:x_cells = ', x_cells
                    y_cells=int(num_cells[1])
                    print 'IN:y_cells = ', y_cells
                elif len(num_cells) == 5:
                    x_cells=int(num_cells[0])
                    print 'IN:x_cells = ', x_cells
                    y_cells=int(num_cells[1])
                    print 'IN:y_cells = ', y_cells
                    z_cells=int(num_cells[2])
                    print 'IN:z_cells = ', z_cells
            if '.history_indices' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                history_indices=lhsrhs[1].split()
                if len(history_indices) == 2:
                    x_index=int(history_indices[0])
                    print 'IN:x_index = ', x_index
                    y_index=int(history_indices[1])
                    print 'IN:y_index = ', y_index
                elif len(history_indices) == 3:
                    x_index=int(history_indices[0])
                    print 'IN:x_index = ', x_index
                    y_index=int(history_indices[1])
                    print 'IN:y_index = ', y_index
                    z_index=int(history_indices[2])
                    print 'IN:z_index = ', z_index


f.closed
print '*****************************************'
####
#read output file

ref_time=0.0

print '********** OUTPUT FILE ******************'
fname=find('slurm-*.out', './')

with open(fname[0], 'r') as f:
    for line in f:
        if line.lstrip().startswith('*'): #skip comment
            continue
        line =line.rstrip() #skip blank line
        if not line:
            continue 
        else: #noncomment line
            strippedline=line
            lhsrhs = strippedline.split(":")
            l=0
            while l<len(lhsrhs): #strip white spaces in lhs
                lhsrhs[l]=lhsrhs[l].rstrip()
                lhsrhs[l]=lhsrhs[l].lstrip()
                l=l+1
            #print type( lhsrhs[0])
            if 'TRANSIT TIME' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_time=float(lhsrhs[1])
            if 'THERMAL SPEED' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_speed=float(lhsrhs[1])
            if 'GYROFREQUENCY' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_gyrofrequency=float(lhsrhs[1])
            if 'GYRORADIUS' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_gyroradius=float(lhsrhs[1])
            if 'DEBYE LENGTH' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_debyelength=float(lhsrhs[1])
            if 'LARMOR NUMBER' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_larmornumber=float(lhsrhs[1])
            if 'DEBYE NUMBER' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_debyenumber=float(lhsrhs[1])
f.closed
print '*****************************************'

print '********** CONSTANTS ********************'
qe=0.00000000048032
me=9.1094E-028
mpn=1.6726E-024
c= 29979000000
print 'qe  [StatC]= ',qe
print 'me  [gram] = ',me
print 'mpn [gram] = ',mpn
print 'c   [cm/s] = ',c

print '*****************************************'


from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy import *
from sympy.abc import x, y, z
from sympy.utilities.lambdify import implemented_function
from sympy import Function

transformations = (standard_transformations + (implicit_multiplication_application,))
pe=parse_expr(n0_grid_func,transformations=transformations)
print 'parsed expression = ',pe

f = lambdify((x,y),pe)
print f(pi,pi) #test

xx = np.linspace(0.0, np.pi*2, 100)
xcm  = np.linspace(0.0, x_max*100, 100)
yy = np.linspace(0.0, np.pi*2, 100)

for i in range(len(xx)):
    yy[i] = f(xx[i],pi)

dx = x_max*100/len(xx) #in [cm]
dyydx = np.gradient(yy)/dx
dlnyydx = dyydx/yy
dlnyydx[0] = dlnyydx[1] #fix discontinuity
dlnyydx[len(dlnyydx)-1] = dlnyydx[len(dlnyydx)-2] #fix discontinuity

print 'average density = ',np.average(yy)

print 'dln(n)/dx minimum = ',min(dlnyydx)


print '********** DERIVED VARS *****************'
b_t = bz_inner*1E4
print 'b_t          [gauss] = ', b_t



c_ion_thermalspeed         = 41900000*((t0_grid_func*units_temperature)**0.5)*((me/mpn/ion_mass)**0.5)
c_elec_thermalspeed        = 41900000*(boltzmann_electron_temperature*units_temperature)**0.5
c_ion_transittimefor100cm  = units_length*100/ c_ion_thermalspeed

c_elec_transittimefor100cm = units_length*100/ c_elec_thermalspeed
c_ion_gyrofrequency        = 9580*(b_t/ ion_mass )
c_elec_gyrofrequency       = 9580*(b_t/ ion_mass *ion_mass )*mpn/ me
c_ion_gyroradius           = c_ion_thermalspeed / c_ion_gyrofrequency
c_elec_gyroradius          = c_elec_thermalspeed / c_elec_gyrofrequency
#c_debyelength=float(lhsrhs[1])
#c_larmornumber=float(lhsrhs[1])
#c_debyenumber=float(lhsrhs[1])


print 'c_ion_thermalspeed      [cm/s] = ', c_ion_thermalspeed, '     (/ref: ', c_ion_thermalspeed/(ref_speed*100),' )' 
print 'c_elec_thermalspeed     [cm/s] = ', c_elec_thermalspeed, '     (/ref: ', c_elec_thermalspeed/(ref_speed*100), ' )'
print 'c_ion_transittimefor100cm  [s] = ', c_ion_transittimefor100cm, ' (/ref: ', c_ion_transittimefor100cm/ref_time,' )'
print 'c_elec_transittimefor100cm [s] = ', c_elec_transittimefor100cm, ' (/ref: ', c_elec_transittimefor100cm/ref_time, ' )'
print 'c_ion_gyrofrequency      [1/s] = ', c_ion_gyrofrequency, '       (/ref: ', c_ion_gyrofrequency/ ref_gyrofrequency, ' )'
print 'c_elec_gyrofrequency     [1/s] = ', c_elec_gyrofrequency, ' (/ref: ', c_elec_gyrofrequency/ ref_gyrofrequency, ' )'
print 'c_ion_gyroradius          [cm] = ', c_ion_gyroradius, '   (/ref: ', c_ion_gyroradius / (units_length*100), ' )'
print 'c_elec_gyroradius         [cm] = ', c_elec_gyroradius, '   (/ref: ', c_elec_gyroradius / (units_length*100), ' )'


k_y        = 2.0*np.pi*m_y/(y_max*100)
deltaL_max = 1./max(abs(dlnyydx))
x_diagnotic_index_in_plot = int(float(x_index)/float(x_cells)*len(dlnyydx))-1 
#print 'x_diagnotic_index_in_plot = ', x_diagnotic_index_in_plot
deltaL_diagnostic = abs(1./(dlnyydx[x_diagnotic_index_in_plot]))
c_s        = 979000*((t0_grid_func*units_temperature)/ion_mass)**0.5
rho_s      = 102*((ion_mass*t0_grid_func*units_temperature)**0.5)/b_t
chi        = k_y*rho_s
omega_star = c_s*rho_s*k_y/deltaL_max
omega_star_diagnostic = c_s*rho_s*k_y/ deltaL_diagnostic

print 'k_y             [1/cm] = ', k_y
print 'deltaL_max        [cm] = ', deltaL_max
print 'deltaL_diagnostic [cm] = ', deltaL_diagnostic
print 'c_s             [cm/s] = ', c_s
print 'rho_s             [cm] = ', rho_s
print 'chi                [-] = ', chi
print 'omega*           [1/s] = ', omega_star
print 'omega*_diagnotic [1/s] = ', omega_star_diagnostic




print '*****************************************'



fig1, ax1 = plt.subplots(2,1)
ax1[0].plot(xcm,yy )
ax1[0].set_xlabel('x (cm)')
ax1[0].set_ylabel('density')
ax1[1].plot(xcm,dlnyydx )
ax1[1].scatter(xcm[x_diagnotic_index_in_plot],dlnyydx[x_diagnotic_index_in_plot] )
ax1[1].set_xlabel('x (cm)')
ax1[1].set_ylabel('d(ln n)/dx')
#ax[1].set_ylim(-2, 0) 
#plt.show()


#read history file
x_list=[]
y_list=[]
with open("potential_hist_1.curve", 'r') as f:
    for line in f:
        if line.lstrip().startswith('#'): #skip comment
            continue
        line =line.rstrip() #skip blank line
        if not line:
            continue 
        else: #noncomment line
            strippedline=line
            lhsrhs = strippedline.split(" ")
            l=0
            while l<len(lhsrhs): #strip white spaces in lhs
                lhsrhs[l]=lhsrhs[l].rstrip()
                lhsrhs[l]=lhsrhs[l].lstrip()
                l=l+1
            x_list.append(float(lhsrhs[0]))
            y_list.append(float(lhsrhs[1]))

f.closed

#del x_list[-10:]
#del y_list[-10:]

#make time unit to second/2/pi
#print type(x_list)
x_list[:] = [i*ref_time/2.0/np.pi for i in x_list] 
#print x_list

# number of signal points
#N = 400
#T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N)
#y = np.exp(50.0 * 1.j * 2.0*np.pi*x) #+ 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)

N = len(x_list)
T = x_list[len(x_list)-1] / N
xt = np.array(x_list)
y = np.array(y_list)

yf = scipy.fftpack.fft(y)
xf = scipy.fftpack.fftfreq(N, T)

xf = scipy.fftpack.fftshift(xf)
yplot = scipy.fftpack.fftshift(yf)

#print np.abs(yplot).argmax()
freqmax=xf[np.abs(yplot).argmax()]
print '|maximum frequency| =', abs(freqmax),'[Hz]'
yv = np.real(y.max()*np.exp(abs(freqmax)*1.j*2.0*np.pi*xt))


yfv = scipy.fftpack.fft(yv)
xfv = scipy.fftpack.fftfreq(N, T)

xfv = scipy.fftpack.fftshift(xfv)
yplotv = scipy.fftpack.fftshift(yfv)


print 'omega_star_computation/omega*             = ', abs(freqmax)/omega_star
print 'omega_star_computation/omega*_diagntostic = ', abs(freqmax)/ omega_star_diagnostic


fig2, ax2 = plt.subplots(2, 1)
ax2[0].plot(xt,y,'xb-')
ax2[0].set_xlabel('Time (s)')
ax2[0].set_ylabel('Amplitude')

ax2[0].plot(xt,yv,'.r-')

ax2[1].plot(xfv,1.0/N * np.abs(yplot),'.b-') # plotting the frequency spectrum
ax2[1].set_xlabel('Freq (Hz)')
ax2[1].set_ylabel('|Y(freq)|')

ax2[1].plot(xfv,1.0/N * np.abs(yplotv),'.r-')

plt.show()

