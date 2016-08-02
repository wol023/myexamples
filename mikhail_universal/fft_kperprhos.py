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
boltzmann_electron_temperature = -1
electron_temperature = -1
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
                #print 'IN:units_number_density = ',units_number_density
            if 'units.temperature' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_temperature = float(lhsrhs[1])
                print 'IN:units_temperature = ',units_temperature
            if 'units.length' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_length = float(lhsrhs[1])
                #print 'IN:units_length= ',units_length
            if 'units.mass' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_mass = float(lhsrhs[1])
                #print 'IN:units_mass = ',units_mass
            if 'units.magnetic_field' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_magnetic_field = float(lhsrhs[1])
                #print 'IN:units_magnetic_field = ',units_magnetic_field
            if 'boltzmann_electron.temperature' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                boltzmann_electron_temperature = float(lhsrhs[1])
                print 'IN:boltzmann_electron_temperature = ',boltzmann_electron_temperature
            if 'gksystem.magnetic_geometry_mapping.slab.Bz_inner' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                bz_inner = float(lhsrhs[1])
                #print 'IN:bz_inner = ',bz_inner
            if 'kinetic_species.1.mass' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                ion_mass = float(lhsrhs[1])
                #print 'IN:ion_mass = ',ion_mass
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
                #print 'IN:x_max = ',x_max
            if 'gksystem.magnetic_geometry_mapping.slab.y_max' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                y_max = float(lhsrhs[1])
                #print 'IN:y_max = ',y_max

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
                #print lhsrhs[0],'=',lhsrhs[1]
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


f.closed
#print '*****************************************'
####
if  boltzmann_electron_temperature == -1:
    electron_temperature=t0_grid_func
    print 'Te(kin) = ', electron_temperature
else:
    electron_temperature = boltzmann_electron_temperature
    print 'Te(bol) = ', electron_temperature
    

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
#print '*****************************************'

#print '********** CONSTANTS ********************'
qe=0.00000000048032
me=9.1094E-028
mpn=1.6726E-024
c= 29979000000
#print 'qe  [StatC]= ',qe
#print 'me  [gram] = ',me
#print 'mpn [gram] = ',mpn
#print 'c   [cm/s] = ',c

#print '*****************************************'


from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy import *
from sympy.abc import x, y, z
from sympy.utilities.lambdify import implemented_function
from sympy import Function

transformations = (standard_transformations + (implicit_multiplication_application,))
pe=parse_expr(n0_grid_func,transformations=transformations)
#print 'parsed expression = ',pe

f = lambdify((x,y,z),pe)
#print f(pi,pi) #test

xx = np.linspace(0.0, np.pi*2, (x_cells))
xcm  = np.linspace(0.0, x_max*100, (x_cells))
yy = np.linspace(0.0, np.pi*2, (x_cells))
yypert = np.linspace(0.0, np.pi*2, (x_cells))

for i in range(len(xx)):
    yy[i] = f(xx[i],0,0)

print 'yy'
print yy

#calc yypert
dphasepert = np.pi/10
#find optimum phase shift
maxyypert_amp = 0.0
maxyypert_phi_ind =0

for j in range(10):
    for i in range(len(xx)):
        yypert[i] =abs(f(xx[i],dphasepert*j,0) - yy[i])
    tempyypert_amp = max( (yypert))
    if tempyypert_amp>maxyypert_amp:
        maxyypert_amp=tempyypert_amp
        maxyypert_phi_ind=j
for i in range(len(xx)):
    yypert[i] =abs(f(xx[i],dphasepert*maxyypert_phi_ind,0) - yy[i])


dx = x_max*100/len(xx) #in [cm]
dyydx = np.gradient(yy)/dx
print 'dyy'
print np.gradient(yy)
print 'dyydx'
print dyydx
dlnyydx = dyydx/yy
print 'dylnydx'
print dlnyydx
dlnyydx[0] = dlnyydx[1] #fix discontinuity
dlnyydx[len(dlnyydx)-1] = dlnyydx[len(dlnyydx)-2] #fix discontinuity

#print 'average density = ',np.average(yy)

#calculate spread indices
contains_tanh =0
contains_pert =0

if 'tanh' in n0_grid_func:
    contains_tanh =1
    #print "contains_tanh"
if ')**2' in n0_grid_func:
    contains_pert =1
    #print "contains_pert"

if contains_tanh and contains_pert:
    dlnyydx_abs = abs(dlnyydx)
    dlnyydx_amplitude= max(dlnyydx_abs)
    ispread_width=sum(dlnyydx_abs>dlnyydx_amplitude*0.33)
    yypert_amplitude = max(yypert)
    ispread_width_density_pert=sum(yypert>yypert_amplitude*0.10)
    ispread_width = min(ispread_width,ispread_width_density_pert)
elif contains_tanh:
    dlnyydx_abs = abs(dlnyydx)
    dlnyydx_amplitude= max(dlnyydx_abs)
    ispread_width=sum(dlnyydx_abs>dlnyydx_amplitude*0.33)
elif contains_pert:
    yypert_amplitude = max(yypert)
    ispread_width=sum(yypert>yypert_amplitude*0.10)
else:
    ispread_width=1
    #print "does not contain spreading"

print '********** DERIVED VARS *****************'
b_t = bz_inner*1E4
print 'b_t          [gauss] = ', b_t



c_ion_thermalspeed         = 41900000*(( t0_grid_func*units_temperature)**0.5)*((me/mpn/ion_mass)**0.5)
c_elec_thermalspeed        = 41900000*(electron_temperature*units_temperature)**0.5
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
k_x        = 2.0*np.pi*0.5/(x_max*100) 
deltaL_max = 1./max(abs(dlnyydx))
x_point_index_in_plot = int(float(x_index)/float(x_cells)*len(dlnyydx))-1 
#print 'x_point_index_in_plot = ', x_point_index_in_plot
deltaL_point= abs(1./(dlnyydx[x_point_index_in_plot]))

deltaL_spread =0.0
#ispread_width = 10
ispread_width = float(ispread_width)
spread_ind_diff = range(-int(np.floor(ispread_width/2)),int(np.ceil(ispread_width/2)))
if int(ispread_width)%2==0:
    tempfirstinddiff=-spread_ind_diff[0]
    del spread_ind_diff[0]
    spread_ind_diff.append(tempfirstinddiff)
#print spread_ind_diff

for ind in spread_ind_diff :
    deltaL_spread = deltaL_spread+dlnyydx[x_point_index_in_plot+ind] 
deltaL_spread = deltaL_spread/ispread_width
deltaL_spread = 1.0/abs(deltaL_spread)
#print type(spread_ind_diff)
spread_ind_diff=np.array(spread_ind_diff)
spread_ind =  spread_ind_diff+ x_point_index_in_plot
#print spread_ind

c_s        = 979000*((electron_temperature*units_temperature)/ion_mass)**0.5
rho_s      = 102*((ion_mass*electron_temperature*units_temperature)**0.5)/b_t
chi        = sqrt(k_x*k_x+k_y*k_y)*rho_s
omega_star = c_s*rho_s*k_y/deltaL_max
omega_star_point= c_s*rho_s*k_y/ deltaL_point
omega_star_spread= c_s*rho_s*k_y/ deltaL_spread

print 'k_y             [1/cm] = ', k_y
print 'k_x             [1/cm] = ', k_x , 'check m_y = (',m_y,') with kinetic.in'
print 'deltaL_max        [cm] = ', deltaL_max
print 'deltaL_point      [cm] = ', deltaL_point
print 'deltaL_spread     [cm] = ', deltaL_spread
print 'c_s             [cm/s] = ', c_s
print 'rho_s             [cm] = ', rho_s
print 'k_y*rho_s          [-] = ', k_y*rho_s
print 'k_y*rho_i          [-] = ', k_y*c_ion_gyroradius
print 'k_perp*rho_s       [-] = ', chi
print 'k_perp*rho_i       [-] = ', chi/rho_s*c_ion_gyroradius
print 'omega*           [1/s] = ', omega_star
print 'omega*_point     [1/s] = ', omega_star_point
print 'omega*_spread    [1/s] = ', omega_star_spread

print '*****************************************'

from setupplot import init_plotting
#### first plot

init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(xcm,yy ,linestyle='-',linewidth=1,color='b',label='density')
plt.scatter(xcm[x_point_index_in_plot],yy[x_point_index_in_plot],marker="o",linewidth=1,color='g',label='measured point' )
plt.xlabel(u'x (cm)')
plt.ylabel(u'density')
#plt.ylabel(r'$\omega_{\mathrm{fit}} / (\omega_*/(1+k_y^2\rho_s^2)) $',fontsize=1.5*plt.rcParams['font.size'])
#plt.title(u'Drift wave frequency')
plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
## output resulting plot to file
#plt.ylim(0.8,1.05)
#
plt.tight_layout()
plt.savefig('foo1.png')
plt.savefig('foo1.eps')
plt.close('all')
#plt.clf()

######
init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(xcm,-dlnyydx,linestyle='-',linewidth=1,color='b',label='inverse gradient length' )
plt.plot(xcm,yypert*10000 ,marker='.',linestyle='-',linewidth=1,color='r',label='perturbationx10000' )
plt.scatter(xcm[spread_ind],-dlnyydx[spread_ind],label='average points' )
plt.xlabel(u'x (cm)')
plt.ylabel(u'perturbation, -d(ln n)/dx [cm]')
plt.gca().legend(bbox_to_anchor = (0.0, 0.15))
plt.tight_layout()
plt.savefig('foo2.png')
plt.savefig('foo2.eps')
plt.close('all')
#plt.clf()

#read history file
x_list=[]
y_list=[]
prev_lhsrhs_0 = '0000'
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
            if  prev_lhsrhs_0 != lhsrhs[0]:
                x_list.append(float(lhsrhs[0]))
                y_list.append(float(lhsrhs[1]))
                prev_lhsrhs_0 = lhsrhs[0]

f.closed

#print x_list

#del x_list[-25:]
#del y_list[-25:]

#print x_list 


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
T = (x_list[len(x_list)-1]-x_list[0]) / N
xt = np.array(x_list)
dimensional_xt = xt*2.0*np.pi
y = np.array(y_list)

yf = scipy.fftpack.fft(y)
xf = scipy.fftpack.fftfreq(N, T)

xf = scipy.fftpack.fftshift(xf)
yplot = scipy.fftpack.fftshift(yf)

#print np.abs(yplot).argmax()


freqmax=xf[np.abs(yplot).argmax()]
print '|maximum  freq. fft| =', abs(freqmax),'[Hz]'

dphase = np.pi/100
#find optimum phase shift
yvydiff2 = np.zeros(100)
for i in range(100):
    phase=dphase*i
    yv = np.real(y.max()*np.exp(abs(freqmax)*1.j*dimensional_xt+1.j*phase) )
    temperror=0.0
    for j in range(len(yv)):
        temperror = temperror+ (yv[j]-y[j])**2
    yvydiff2[i] = temperror

minphaseind=np.abs(yvydiff2).argmin()
#print minphaseind
yv = np.real(y.max()*np.exp(abs(freqmax)*1.j*dimensional_xt+1.j*dphase*minphaseind) )


yfv = scipy.fftpack.fft(yv)
xfv = scipy.fftpack.fftfreq(N, T)

xfv = scipy.fftpack.fftshift(xfv)
yplotv = scipy.fftpack.fftshift(yfv)


################# leastsq fitting
from scipy.optimize import leastsq

#data = 3.0*np.sin(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

guess_amplitude = (y.max()-y.min())/2
guess_mean = np.mean(y)
guess_phase = 0
guess_freq = freqmax
guess_lin = 0

# we'll use this to plot our first estimate. This might already be good enough for you
#data_first_guess = guess_amplitude*np.cos(guess_freq*dimensional_xt+guess_phase)+guess_mean

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda z: z[3]*np.cos(z[0]*dimensional_xt+z[1]) + z[2] - y
#optimize_func = lambda z: guess_amplitude*np.cos(z[0]*dimensional_xt+z[1]) + z[2] - y
est_freq, est_phase, est_mean, est_amplitude = leastsq(optimize_func, [guess_freq, guess_phase, guess_mean, guess_amplitude ])[0]

data_fit = est_mean + est_amplitude*np.cos(est_freq*dimensional_xt+est_phase) 
# recreate the fitted curve using the optimized parameters

#guess_amplitude = guess_amplitude-est_lin*(xt[len(xt)-1]-xt[0])
#guess_freq = freqmax+4E5
#
#data_second_guess = guess_mean+est_lin*xt + guess_amplitude*np.cos(guess_freq*2.0*np.pi*xt+guess_phase) 
#
#optimize_func = lambda z: z[3]*np.cos(z[0]*2.0*np.pi*xt+z[1]) + z[2] - (y -est_lin*xt)
#est_freq, est_phase, est_mean, est_amplitude = leastsq(optimize_func, [guess_freq, guess_phase, guess_mean, guess_amplitude])[0]
#
#data_fit = est_mean+est_lin*xt + est_amplitude*np.cos(est_freq*2.0*np.pi*xt+est_phase) 

print '|optimized  freq. fitting| =', abs(est_freq),'[Hz]'


yfv_fit = scipy.fftpack.fft(data_fit)
xfv_fit = scipy.fftpack.fftfreq(N, T)

xfv_fit = scipy.fftpack.fftshift(xfv_fit)
yplotv_fit = scipy.fftpack.fftshift(yfv_fit)






############




init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(dimensional_xt,y,marker='.',linestyle='-',linewidth=1,color='b',label='potential' )
plt.plot(dimensional_xt,yv,linestyle='-',linewidth=1,color='r',label='fourier mode' )
plt.plot(dimensional_xt,data_fit,marker='.',linestyle='-',linewidth=1,color='g',label='fitting' )
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlabel(u'Time (s)')
plt.ylabel(u'Amplitude')
plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
plt.tight_layout()
plt.savefig('foo3.png')
plt.savefig('foo3.eps')
plt.close('all')
#plt.clf()



init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(xf,1.0/N*np.abs(yplot),marker='.',linestyle='-',linewidth=1,color='b',label='frequency spectrum' )
#plt.plot(xf,1.0/N*np.abs(yplotv),linestyle='-',linewidth=1,color='r',label='dominant spectrum' )
plt.plot(xf,1.0/N*np.abs(yplotv_fit),marker='.',linestyle='-',linewidth=1,color='g',label='fitted frequency spectrum' )
xf2lim=xf[len(xf)/2+abs( len(xf)/2-np.argmax(abs(yplotv_fit)) )*3]
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlim(-abs(xf2lim),abs(xf2lim))
#plt.xlim(plt.gca().get_xlim()[0]*0.1,plt.gca().get_xlim()[1]*0.1)
plt.ylim(plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]*1.3)
plt.xlabel(u'Freq. (Hz)')
plt.ylabel(u'|Y[Freq.]|')
plt.gca().legend(bbox_to_anchor = (0.1, 0.9))
plt.tight_layout()
plt.savefig('foo4.png')
plt.savefig('foo4.eps')
plt.close('all')
#plt.clf()

#take log
logy2 = np.log(y*y)
#find zero crossing
#zero_crossings = np.where(np.diff(np.sign(np.gradient(logy2))))[0]
zero_crossings = np.where(np.gradient(np.sign(np.gradient(logy2)))<0 )[0]
extremum_dimensional_xt = dimensional_xt[zero_crossings]
extremum_logy2 = logy2[zero_crossings]
lin_fitted_logy2 = np.polyfit(extremum_dimensional_xt,extremum_logy2,1)
legend_lin_fitted_logy2 = 'y = %g x + %g' % (lin_fitted_logy2[0],lin_fitted_logy2[1])
print legend_lin_fitted_logy2

init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(dimensional_xt,logy2,linestyle='-',linewidth=1,color='b',label='ln(phi^2)' )
plt.plot(extremum_dimensional_xt,extremum_logy2,marker='x',linewidth=1,color='g',label='extremum points' )
plt.plot(extremum_dimensional_xt,lin_fitted_logy2[0]*extremum_dimensional_xt+lin_fitted_logy2[1],color='r',linewidth=1,label=legend_lin_fitted_logy2 )
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlabel(u'Time (s)')
plt.ylabel(u'ln(|phi|^2)')
plt.gca().legend(bbox_to_anchor = (0.1, 0.3))
plt.tight_layout()
plt.savefig('foo5.png')
plt.savefig('foo5.eps')
plt.close('all')
#plt.clf()

starting_amplitude= np.exp((extremum_logy2[-1]-lin_fitted_logy2[0]*extremum_dimensional_xt[-1])/2.0)
#data_fit_with_growth = est_mean + starting_amplitude*np.exp(lin_fitted_logy2[0]/2.0*dimensional_xt)*np.cos(est_freq*dimensional_xt+est_phase) 
#adjust phase
#data_fit_with_growth_guess = est_mean + starting_amplitude*np.exp(lin_fitted_logy2[0]/2.0*dimensional_xt)*np.cos(est_freq*dimensional_xt+est_phase) 
optimize_func = lambda z: z[1] + starting_amplitude*np.exp(lin_fitted_logy2[0]/2.0*dimensional_xt)*np.cos(est_freq*dimensional_xt+z[0]) - y
refine_est_phase, refine_est_mean = leastsq(optimize_func, [est_phase, est_mean])[0]
data_fit_with_growth_refine = refine_est_mean + starting_amplitude*np.exp(lin_fitted_logy2[0]/2.0*dimensional_xt)*np.cos(est_freq*dimensional_xt+refine_est_phase) 


legend_data_fit_with_growth = r'$\gamma/\omega^*, \gamma/\omega^*_d$'+' = (%g, %g)' % ( lin_fitted_logy2[0]/2.0/omega_star, lin_fitted_logy2[0]/2.0/omega_star_point*(1.0+chi*chi))
init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(dimensional_xt,y,marker='.',linestyle='-',linewidth=1,color='b',label='potential' )
#plt.plot(dimensional_xt,data_fit_with_growth,linestyle='-',linewidth=1,color='g',label=legend_data_fit_with_growth )
plt.plot(dimensional_xt,data_fit_with_growth_refine,linestyle='-',linewidth=1,color='r',label=legend_data_fit_with_growth )
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlabel(u'Time (s)')
plt.ylabel(u'phi')
plt.gca().legend(bbox_to_anchor = (0.0, 0.2))
plt.tight_layout()
plt.savefig('foo6.png')
plt.savefig('foo6.eps')
plt.close('all')
#plt.clf()



with open('finish.txt', 'wb') as fh:
    buf = "te = %f\n" % (units_temperature*electron_temperature)
    fh.write(buf)
    buf = "ti = %f\n" % (units_temperature*t0_grid_func)
    fh.write(buf)
    buf = "omega_star = %f\n" % (omega_star)
    fh.write(buf)
    buf = "omega_star_point = %f\n" % (omega_star_point)
    fh.write(buf)
    buf = "omega_star_spread = %f\n" % (omega_star_spread)
    fh.write(buf)
    buf = 'omega_star_fit/omega*        = %f\n'%( abs(est_freq)/omega_star )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_point  = %f\n'%( abs(est_freq)/omega_star_point )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_spread = %f\n'%( abs(est_freq)/omega_star_spread )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_1_chi2  = %f\n'%( abs(est_freq)/omega_star_point*(1.0+chi*chi) )
    fh.write(buf)
    buf = 'gamma  = %f\n'%(lin_fitted_logy2[0]/2.0)
    fh.write(buf)
    buf = 'gamma/omega*  = %f\n'%(lin_fitted_logy2[0]/2.0/omega_star)
    fh.write(buf)
    buf = 'gamma/omega*_1_chi2  = %f\n'%(lin_fitted_logy2[0]/2.0/omega_star_point*(1.0+chi*chi))
    fh.write(buf)


print "te = " , (units_temperature*electron_temperature)
print "ti = " , (units_temperature*t0_grid_func)
print "omega_star = " , (omega_star)
print "omega_star_point = " , (omega_star_point)
print "omega_star_spread = " , (omega_star_spread)
print 'omega_star_FFT/omega*        = ', abs(freqmax)/omega_star
print 'omega_star_FFT/omega*_point  = ', abs(freqmax)/ omega_star_point
print 'omega_star_fit/omega*        = ',( abs(est_freq)/omega_star )
print 'omega_star_fit/omega*_point  = ',( abs(est_freq)/omega_star_point )
print 'omega_star_fit/omega*_spread = ',( abs(est_freq)/omega_star_spread )
print 'omega_star_fit/omega*_point_1_chi2  = ',( abs(est_freq)/omega_star_point*(1.0+chi*chi) )
print 'gamma  = ',(lin_fitted_logy2[0]/2.0)
print 'gamma/omega*  = ',(lin_fitted_logy2[0]/2.0/omega_star)
print 'gamma/omega*_1_chi2  = ',(lin_fitted_logy2[0]/2.0/omega_star_point*(1.0+chi*chi))






import pylab
import Image
init_plotting('2x3')
f = pylab.figure()
for n, fname in enumerate(('foo1.png', 'foo2.png', 'foo3.png', 'foo4.png', 'foo5.png','foo6.png')):
     image=Image.open(fname)#.convert("L")
     arr=np.asarray(image)
     ax=f.add_subplot(2, 3, n+1)
     ax.axis('off')
     pylab.imshow(arr)
pylab.show()



