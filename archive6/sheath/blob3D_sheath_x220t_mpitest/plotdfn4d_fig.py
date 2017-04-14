import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import ndimage
import pylab as pl
from scipy.optimize import leastsq


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
 


# import dfn
File =h5py.File('./plt_dfn_plots/plt.2.electron.dfn0000.4d.hdf5','r')     

#print File.items()
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

cells = File['level_0']['boxes'][0]
print cells
dim=len(cells)/2
print dim
if dim<5:
    xcell_beg=cells[0]
    xcell_fin=cells[0+dim]
    num_xcell=xcell_fin-xcell_beg+1
    print xcell_beg,xcell_fin,num_xcell
    print
    ycell_beg=cells[1]
    ycell_fin=cells[1+dim]
    num_ycell=ycell_fin-ycell_beg+1
    print ycell_beg,ycell_fin,num_ycell
    print
    vparcell_beg=cells[2]
    vparcell_fin=cells[2+dim]
    num_vparcell=vparcell_fin-vparcell_beg+1
    print vparcell_beg,vparcell_fin,num_vparcell
    print
    mucell_beg=cells[3]
    mucell_fin=cells[3+dim]
    num_mucell=mucell_fin-mucell_beg+1
    print mucell_beg,mucell_fin,num_mucell
else:
    xcell_beg=cells[0]
    xcell_fin=cells[0+dim]
    num_xcell=xcell_fin-xcell_beg+1
    print xcell_beg,xcell_fin,num_xcell
    print
    ycell_beg=cells[1]
    ycell_fin=cells[1+dim]
    num_ycell=ycell_fin-ycell_beg+1
    print ycell_beg,ycell_fin,num_ycell
    print
    zcell_beg=cells[2]
    zcell_fin=cells[2+dim]
    num_zcell=zcell_fin-zcell_beg+1
    print zcell_beg,zcell_fin,num_zcell
    print
    vparcell_beg=cells[3]
    vparcell_fin=cells[3+dim]
    num_vparcell=vparcell_fin-vparcell_beg+1
    print vparcell_beg,vparcell_fin,num_vparcell
    print
    mucell_beg=cells[4]
    mucell_fin=cells[4+dim]
    num_mucell=mucell_fin-mucell_beg+1
    print mucell_beg,mucell_fin,num_mucell


data = File['level_0']['data:datatype=0'][:]
print len(data)
data4d=data.reshape((num_xcell,num_ycell,num_vparcell,num_mucell),order='F')
print data4d.shape
File.close()


# diagnostic points
x_pt = 10
y_pt = 10
z_pt = 1
x_rbf_refine = 5
y_rbf_refine = 5
rbf_smooth = 0

#phase
VPAR,MU = np.mgrid[-1:1:(num_vparcell*1j),0:1:(num_mucell*1j)]

#refine
rbf = interpolate.Rbf(VPAR.ravel(), MU.ravel(), data4d[x_pt,y_pt,:,:].ravel(), smooth=rbf_smooth)
VPAR_REFINE, MU_REFINE = np.mgrid[-1:1:(num_vparcell*1j*x_rbf_refine), 0:1:(num_mucell*1j*y_rbf_refine)]
saved_rbf=rbf(VPAR_REFINE,MU_REFINE)


#reconstruct from model
Bhat = 3.0
That = 1.0
mhat = 2.0
nhat = 1.0
Vpar_max = 4
Mu_max = 2

vpar_cell_dim_begin = -1.0*Vpar_max+(Vpar_max*1.0+Vpar_max*1.0)/num_vparcell/2.0
vpar_cell_dim_end   =  1.0*Vpar_max-(Vpar_max*1.0+Vpar_max*1.0)/num_vparcell/2.0
mu_cell_dim_begin = Mu_max*0.0+(Mu_max*1.0-Mu_max*0.0)/num_mucell/2.0
mu_cell_dim_end   = Mu_max*1.0-(Mu_max*1.0-Mu_max*0.0)/num_mucell/2.0
VPAR_CELL,MU_CELL = np.mgrid[vpar_cell_dim_begin:vpar_cell_dim_end:(num_vparcell*1j),mu_cell_dim_begin:mu_cell_dim_end:(num_mucell*1j)]

#VPAR_SCALE = VPAR_CELL[:,0]*np.sqrt(mhat) #for trunk
VPAR_SCALE = VPAR_CELL[:,0] #for mass dependent normalization
MU_SCALE = MU_CELL[0,:]

coef_maxwell=nhat/np.sqrt(np.pi)*(0.5*mhat/That)**(1.5)
f_model = np.zeros((num_vparcell,num_mucell))
for i in range(num_vparcell):
    for j in range(num_mucell):
        f_model[i][j]=coef_maxwell*np.exp(-0.5*(VPAR_SCALE[i]**2+MU_SCALE[j]*Bhat)/That)

delta_f_model = (f_model-data4d[x_pt,y_pt,:,:])

#least square fitting on a slice of MU index=0
guess_den =coef_maxwell 
guess_temp =1.0/(2.0*That)
guess_shift =0.0
optimize_func = lambda z: z[0]*np.exp(-z[1]*(VPAR_SCALE-z[2])**2)-data4d[x_pt,y_pt,:,0] 
est_den, est_temp, est_shift = leastsq(optimize_func, [guess_den, guess_temp, guess_shift])[0]
fitted_f = est_den*np.exp(-est_temp*VPAR_SCALE**2)
t_fit = 1.0/(est_temp*2.0)
n_fit = est_den*np.sqrt(np.pi)/(0.5*mhat/t_fit)**(1.5)/np.exp(-MU_SCALE[0]*Bhat/2.0/t_fit)
vshift_fit = est_shift
print 't_fit = ',t_fit
print 'n_fit = ',n_fit
print 'vshift_fit = ',est_shift


#density sum
X,Y = np.mgrid[0:1:(num_xcell*1j),0:1:(num_ycell*1j)]
f_vpar_mu_sum = np.zeros((num_xcell,num_ycell))
delta_Mu = 1.0*Mu_max/num_mucell
delta_Vpar = 2.0*Vpar_max/num_vparcell
print delta_Mu
print delta_Vpar
for i in range(num_xcell):
    for j in range(num_ycell):
        sumovervparandmu=0.0
        for k in range(num_vparcell):
            sumovermu=0.0
            for l in range(num_mucell):
                sumovermu=sumovermu+data4d[i,j,k,l]
            sumovermu=sumovermu*delta_Mu
            sumovervparandmu=sumovervparandmu+sumovermu
        sumovervparandmu=sumovervparandmu*delta_Vpar 
        f_vpar_mu_sum[i,j]=f_vpar_mu_sum[i,j]+sumovervparandmu


#sum over mu
f_mu_sum = np.zeros(num_vparcell)
for i in range(num_vparcell):
    for j in range(num_mucell):
        f_mu_sum[i]=f_mu_sum[i]+data4d[x_pt,y_pt,i,j]
    f_mu_sum[i]=f_mu_sum[i]/num_mucell

f_mu_sum_rbf = np.zeros(len(VPAR_REFINE[:,0]))
for i in range(len(VPAR_REFINE[:,0])):
    for j in range(len(VPAR_REFINE[0,:])):
        f_mu_sum_rbf[i]=f_mu_sum_rbf[i]+saved_rbf[i,j]
    f_mu_sum_rbf[i]=f_mu_sum_rbf[i]/len(VPAR_REFINE[0,:])




#raw density
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
plt.contourf(X,Y,f_vpar_mu_sum,100)
plt.title(u'n(X,Y)')
plt.xlabel(u'X')
plt.ylabel(u'Y')
plt.tight_layout()
plt.savefig('fig1.png')
plt.savefig('fig1.eps')
plt.close('all')

#raw f
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
plt.contourf(VPAR,MU,data4d[x_pt,y_pt,:,:],100)
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
plt.tight_layout()
plt.colorbar();
plt.savefig('fig2.png')
plt.savefig('fig2.eps')
plt.close('all')

##refine
init_plotting()
fig=plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
plt.contourf(VPAR_REFINE, MU_REFINE, saved_rbf,100)
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
plt.tight_layout()
plt.colorbar();
plt.savefig('fig3.png')
plt.savefig('fig3.eps')
plt.close('all')

#diagnostic point
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
plt.savefig('fig4.png')
plt.savefig('fig4.eps')
plt.close('all')

#summed maxwellian 
init_plotting()
fig=plt.subplot(111)
plt.plot(VPAR[:,0], f_mu_sum )
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel('<f>')
plt.tight_layout()
plt.savefig('fig5.png')
plt.savefig('fig5.eps')
plt.close('all')

#summed maxwellian on refined 
init_plotting()
fig=plt.subplot(111)
plt.plot(VPAR_REFINE[:,0], f_mu_sum_rbf )
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel('<f>')
plt.tight_layout()
plt.savefig('fig6.png')
plt.savefig('fig6.eps')
plt.close('all')

##model maxwellian
init_plotting()
fig=plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
#plt.contourf(VPAR_CELL, MU_CELL, f_model,100)
plt.contourf(VPAR, MU, f_model,100)
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
plt.tight_layout()
plt.colorbar();
plt.savefig('fig7.png')
plt.savefig('fig7.eps')
plt.close('all')

##model maxwellian delta
init_plotting()
fig=plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
plt.contourf(VPAR, MU, delta_f_model,100)
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
plt.tight_layout()
plt.colorbar();
plt.savefig('fig8.png')
plt.savefig('fig8.eps')
plt.close('all')

#compare mu slice with model
init_plotting()
fig=plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
plt.plot(VPAR[:,0], f_model[:,0]/data4d[x_pt,y_pt,:,0])
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(u'f_model/f(MU=0)')
plt.tight_layout()
plt.savefig('fig9.png')
plt.savefig('fig9.eps')
plt.close('all')

#compare mu slice with fitting
legend_maxwellian = 'MAXWELLIAN\n'+r'$n, T, V_s$'+' = (%.1f, %.1f, %.1g)'%(n_fit,t_fit,vshift_fit)
init_plotting()
fig=plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
#plt.plot(VPAR[:,0], fitted_f/data4d[x_pt,y_pt,:,0])
plt.plot(VPAR[:,0], data4d[x_pt,y_pt,:,0],linewidth=1.5,color='b',label='COGENT')
plt.plot(VPAR[:,0], fitted_f,linewidth=1.5,linestyle='--',color='k',label=legend_maxwellian)
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(u'f_fit/f(MU=0)')
#plt.gca().legend(bbox_to_anchor = (0.0, 0.2))
plt.gca().legend(loc='best')
plt.tight_layout()
plt.savefig('fig10.png')
plt.savefig('fig10.eps')
plt.close('all')

#raw f no interpolation
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
plt.imshow(data4d[x_pt,y_pt,:,:].T,interpolation='none',origin="lower",extent=[-1,1,0,1], aspect=1.60)
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
plt.tight_layout()
plt.colorbar();
plt.savefig('fig11.png')
plt.savefig('fig11.eps')
plt.close('all')




#collect figures
from PIL import Image
init_plotting('2x3')
f = pl.figure()
for n, fname in enumerate(('fig1.png','fig2.png','fig11.png','fig4.png','fig7.png','fig10.png')):
     image=Image.open(fname)#.convert("L")
     arr=np.asarray(image)
     ax=f.add_subplot(2, 3, n+1)
     ax.axis('off')
     pl.imshow(arr)
pl.tight_layout()
pl.savefig('fig0.png')
pl.show()





