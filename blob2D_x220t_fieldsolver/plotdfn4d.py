import numpy as np
import h5py
import matplotlib as plt
from scipy import interpolate
from scipy import ndimage
import pylab as pl


File =h5py.File('./plt_dfn_plots/plt.2.electron.dfn0000.4d.hdf5','r')     

print File.items()
print 
print File['Chombo_global'].attrs['SpaceDim']
print
print File['level_0'].items()
print
print File['level_0']['Processors'][:]
print 
print File['level_0']['boxes'][:]
print 
print File['level_0']['data:offsets=0'][:]
print 
print len(File['level_0']['data:datatype=0'][:])
print 
print File['level_0']['data_attributes'].attrs.items()

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


########## multi plots 
fig, axes = pl.subplots(2,3,figsize=(16,8))
#density sum
X,Y = np.mgrid[0:1:(num_xcell*1j),0:1:(num_ycell*1j)]
f_vpar_mu_sum = np.zeros((num_xcell,num_ycell))
for i in range(num_xcell):
    for j in range(num_ycell):
        sumovervparandmu=0.0
        for k in range(num_vparcell):
            sumovermu=0.0
            for l in range(num_mucell):
                sumovermu=sumovermu+data4d[i,j,k,l]
            sumovermu=sumovermu/num_mucell #make independent on number of cells
            sumovervparandmu=sumovervparandmu+sumovermu
        sumovervparandmu=sumovervparandmu/num_vparcell #make independent on number of cells

        f_vpar_mu_sum[i,j]=f_vpar_mu_sum[i,j]+sumovervparandmu

c00 = axes[0][0].contourf(X,Y,f_vpar_mu_sum,100)
pl.colorbar(c00, ax=axes[0][0]);

#diagnostic point
c100 = axes[1][0].contourf(X,Y,f_vpar_mu_sum)
c101 = axes[1][0].scatter(float(x_pt)/num_xcell,float(y_pt)/num_ycell)
pl.colorbar(c100, ax=axes[1][0]);
axes[1][0].set_xlim([0.0,1.0]);
axes[1][0].set_ylim([0.0,1.0]);



#raw f
VPAR,MU = np.mgrid[0:1:(num_vparcell*1j),0:1:(num_mucell*1j)]
c01 = axes[0][1].contourf(VPAR,MU,data4d[x_pt,y_pt,:,:],100)
pl.colorbar(c01, ax=axes[0][1]);

##refine
rbf = interpolate.Rbf(VPAR.ravel(), MU.ravel(), data4d[x_pt,y_pt,:,:].ravel(), smooth=rbf_smooth)
VPAR_REFINE, MU_REFINE = np.mgrid[0:1:(num_vparcell*1j*x_rbf_refine), 0:1:(num_mucell*1j*y_rbf_refine)]
saved_rbf=rbf(VPAR_REFINE,MU_REFINE)
c02 = axes[0][2].contourf(VPAR_REFINE, MU_REFINE, saved_rbf,100)
pl.colorbar(c02, ax=axes[0][2]);

print saved_rbf.shape


#sum over mu
f_mu_sum = np.zeros(num_vparcell)
for i in range(num_vparcell):
    for j in range(num_mucell):
        f_mu_sum[i]=f_mu_sum[i]+data4d[x_pt,y_pt,i,j]
    f_mu_sum[i]=f_mu_sum[i]/num_mucell

f_mu_sum_rbf = np.zeros(len(VPAR_REFINE[:,0]))
print len(VPAR_REFINE[0,:])
for i in range(len(VPAR_REFINE[:,0])):
    for j in range(len(VPAR_REFINE[0,:])):
        f_mu_sum_rbf[i]=f_mu_sum_rbf[i]+saved_rbf[i,j]
    f_mu_sum_rbf[i]=f_mu_sum_rbf[i]/len(VPAR_REFINE[0,:])


c110 = axes[1][1].plot(VPAR[:,0], data4d[x_pt,y_pt,:,0] )
c111 = axes[1][1].plot(VPAR[:,0], f_mu_sum )
c120 = axes[1][2].plot(VPAR_REFINE[:,0], saved_rbf[:,0] )
c121 = axes[1][2].plot(VPAR_REFINE[:,0], f_mu_sum_rbf )

pl.show()





    


