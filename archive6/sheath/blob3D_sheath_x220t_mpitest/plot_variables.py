import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import ndimage

import pylab as pl

#for least squre fit
from scipy.optimize import leastsq

#to 3D plot
from mayavi import mlab

#for sleep
import time

#############################################################
########################## diagnostic points
x_pt = 3
y_pt = 4
z_pt = 7
#############################################################
#############################################################
from plot_cogent_pack import *

#############################
###############import dfn
targetfilename='./plt_dfn_plots/plt.2.electron.dfn0000.5d.hdf5'
if 'dfn' in targetfilename:
    plot_dfn(targetfilename,x_pt=x_pt,y_pt=y_pt,z_pt=z_pt)

#############################
###############import potential
targetfilename='./plt_potential_plots/plt.potential0001.3d.hdf5'
if 'potential' in targetfilename:
    plot_potential(targetfilename,ghost=0,x_slice=0.5,y_slice=0.5,z_slice=0.5)

#############################
###############import Bvector
targetfilename='./BField_cc3d.hdf5'
if 'BField' in targetfilename:
    plot_bvec(targetfilename,ghost=0)

#############################
###############import Evector
targetfilename='./plt_efield_plots/plt.efield0001.cell.3d.hdf5'
if 'efield' in targetfilename:
    plot_evec(targetfilename,ghost=0)

#############################
###############import density
targetfilename='./plt_density_plots/plt.1.hydrogen.density0001.3d.hdf5'
if 'density' in targetfilename:
    plot_density(targetfilename,ghost=0)

#collect figures

figure_list=os.listdir('./python_auto_plots')
demo_list_full=[]
for i in range(len(figure_list)):
    if 'png' in figure_list[i]:
        demo_list_full.append('./python_auto_plots/'+figure_list[i]);

print demo_list_full

import fnmatch

demo_list=[d_list for d_list in demo_list_full if not fnmatch.fnmatch(d_list,'*/fig[0-9].png')]
demo_list.sort()


canverse_num = len(demo_list)/6
if len(demo_list)%6:
    canverse_num+=1

from PIL import Image

for num_can in range(canverse_num):
    init_plotting('2x3')
    f = pl.figure()
    for n, fname in enumerate(demo_list[num_can*6:min(num_can*6+6,len(demo_list))]):
         print n, fname
         image=Image.open(fname)#.convert("L")
         arr=np.asarray(image)
         ax=f.add_subplot(2, 3, n+1)
         ax.axis('off')
         pl.imshow(arr)
    pl.tight_layout()
    demo_sum='./python_auto_plots/fig'+'%d'%num_can+'.png'
    pl.savefig(demo_sum)
    print demo_sum, ' written.'
pl.show()






