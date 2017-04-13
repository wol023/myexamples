import numpy as np
import h5py
import matplotlib as plt
from scipy import interpolate
from scipy import ndimage
import pylab as pl


File =h5py.File('./plt_density_plots/plt.2.electron.density0000.2d.hdf5','r')     
#print

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
#print File['level_0']['data:datatype=0'][:]
#print 
#print File['level_0']['data_attributes'].attrs.items()

cells = File['level_0']['boxes'][:]
print type(cells)
print cells[0]
#for cell in cells[0]:
#    print cell
for i in range(len(cells[0])):
    print cells[0][i]
xindex=len(cells[0])/2
yindex=len(cells[0])/2+1
xcell_fin=cells[0][xindex]
ycell_fin=cells[0][yindex]
print 'xcell_fin=',xcell_fin
print 'ycell_fin=',ycell_fin
num_xcell=xcell_fin+1
num_ycell=ycell_fin+1

data = File['level_0']['data:datatype=0'][:]
print type(data)
print len(data)

#data2d=np.zeros((num_xcell,num_ycell))
#for i in range(len(data)):
#    data2d[i%num_xcell][i/num_xcell]=data[i]
data2d=data.reshape((num_xcell,num_ycell),order='F')

File.close()

print data2d.shape

X,Y = np.mgrid[0:1:(num_xcell*1j),0:1:(num_ycell*1j)]
fig, axes = pl.subplots(1,3,figsize=(15,4))
c1 = axes[0].contourf(X,Y,data2d)
pl.colorbar(c1, ax=axes[0]);

#tmp = np.repeat(np.repeat(data2d, 10, axis=1), 10, axis=0)
#X2, Y2 = np.mgrid[0:1:(num_xcell*1j*10), 0:1:(num_ycell*1j*10)]
#c2 = axes[1].contourf(X2, Y2, ndimage.gaussian_filter(tmp, 3), levels=c1.levels)
#pl.colorbar(c2, ax=axes[1]);

rbf = interpolate.Rbf(X.ravel(), Y.ravel(), data2d.ravel(), smooth=0)
X3, Y3 = np.mgrid[0:1:(num_xcell*1j*10), 0:1:(num_ycell*1j*10)]
c3 = pl.contourf(X3, Y3, rbf(X3, Y3))
pl.colorbar(c3, ax=axes[2]);

pl.show()






    


        







    


