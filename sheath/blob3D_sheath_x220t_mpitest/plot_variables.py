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
dfnfilename='./plt_dfn_plots/plt.2.electron.dfn0000.5d.hdf5'
dataNd_dfn_comps=import_multdim_comps(filename=dfnfilename)
title_var='distr. function'
num_cell_total_comps_tuple=dataNd_dfn_comps.shape
from read_input_deck import *
VPAR_SCALE, MU_SCALE = get_vpar_mu_scales(num_cell_total_comps_tuple,Vpar_max,Mu_max)
VPAR_N, MU_N = get_vpar_mu_scales(num_cell_total_comps_tuple)

#velocity space
fig_dfn2d=plot_Nd(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],title=title_var)
plt.savefig('fig_vpar_mu.png')
plt.savefig('fig_vpar_mu.eps')
plt.close('all')

fig_dfn2d_interp=plot_Nd(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],title=title_var,interpolation='spline36')
plt.savefig('fig_vpar_mu_interp.png')
plt.savefig('fig_vpar_mu_interp.eps')
plt.close('all')
#plt.close(fig_dfn2d)

#example vpar plot overplot
fig=oplot_1d(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,0,:],xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,0,:])),label='COGENT'%MU_N[0],legend=1 )
#oplot_1d(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,1,:],fig=fig,xaxis=np.linspace(-1,1,len(dataNd_dfn_comps[x_pt,y_pt,z_pt,:,1,:])),label='f(mu=%g)'%(MU_N[1]),legend=1 )
#example mu=0 fitting
coef_maxwell,mhat,That = get_maxwellian_coef(dfnfilename,ion_mass,t0_grid_func,elec_mass,et0_grid_func,nhat)
fitted_f, n_fit, t_fit, vshift_fit = get_maxwellian_fitting(coef_maxwell,mhat, That,Bhat,dataNd_dfn_comps[x_pt,y_pt,z_pt,:,:,:],VPAR_SCALE,MU_SCALE,mu_ind=0)
# plot fitting
legend_maxwellian = '\n\nMAXWELLIAN\n'+r'$n, T, V_s$'+' = (%.1f, %.1f, %.1g)'%(n_fit,t_fit,vshift_fit)
oplot_1d(fitted_f,fig,title='',linewidth=1.5, linestyle='--',color='k',label=legend_maxwellian,legend=1)
plt.savefig('fig_vpar_compare.png')
plt.savefig('fig_vpar_compare.eps')
plt.close('all')

#get density by summation over velocity
f_vpar_mu_sum = get_summation_over_velocity(dataNd_dfn_comps,Vpar_max,Mu_max)
fig=plot_Nd(f_vpar_mu_sum,title='integrated f')
mlab.savefig('fig_f_sum_mlab_iso.png')
fig.scene.save_ps('fig_f_sum_mlab_iso.eps')
#fig.scene.save_ps('fig1_mlab_iso.pdf')
#mlab.show()
#arr=mlab.screenshot()
#plt.imshow(arr)
#plt.axis('off')
#plt.savefig('fig1_mlab_iso.eps')
#plt.savefig('fig1_mlab_iso.pdf')
#fig.scene.close()
mlab.close(all=True)
#sliced plot
fig=plot_Nd(f_vpar_mu_sum,title='integrated f',x_slice=2,y_slice=2,z_slice=2)
mlab.savefig('fig_f_sum_mlab_slice.png')
fig.scene.save_ps('fig_f_sum_mlab_slice.eps')
time.sleep(1)
mlab.close(all=True)
 




#############################
###############import potential
# import potential
potentialfilename='./plt_potential_plots/plt.potential0001.3d.hdf5'
dataNd_potential_comps=import_multdim_comps(filename=potentialfilename)
dataNd_potential_with_outer_ghost_comps,num_ghost_potential=import_multdim_comps(filename=potentialfilename,withghost=1)
title_var='potential'

fig=plot_Nd(dataNd_potential_comps,title=title_var)
mlab.savefig('fig_potential_mlab_iso.png')
fig.scene.save_ps('fig_potential_mlab_iso.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_potential_with_outer_ghost_comps,num_ghost_potential,title=title_var)
mlab.savefig('fig_potential_mlab_ghost_iso.png')
fig.scene.save_ps('fig_potential_mlab_ghost_iso.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_potential_comps,title=title_var,sliced=1,x_slice=0.5,y_slice=0.5,z_slice=0.5)
mlab.savefig('fig_potential_mlab_slice.png')
fig.scene.save_ps('fig_potential_mlab_slice.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_potential_with_outer_ghost_comps,num_ghost_potential,title=title_var,x_slice=0.5,y_slice=0.5,z_slice=0.5)
mlab.savefig('fig_potential_mlab_ghost_slice.png')
fig.scene.save_ps('fig_potential_mlab_ghost_slice.eps')
time.sleep(1)
mlab.close(all=True)

#############################
###############import Bvector
# import B bector
bvecfilename='./BField_cc3d.hdf5'
dataNd_bvec_comps=import_multdim_comps(filename=bvecfilename)
dataNd_bvec_with_outer_ghost_comps,num_ghost_bvec =import_multdim_comps(filename=bvecfilename,withghost=1)
title_var='B field'

fig=plot_Nd(dataNd_bvec_comps,title=title_var)
mlab.savefig('fig_bvec_mlab.png')
fig.scene.save_ps('fig_bvec_mlab.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_bvec_with_outer_ghost_comps,num_ghost_bvec,title=title_var)
mlab.savefig('fig_bvec_mlab_ghost.png')
fig.scene.save_ps('fig_bvec_mlab_ghost.eps')
time.sleep(1)
mlab.close(all=True)

#############################
###############import Evector
evecfilename='./plt_efield_plots/plt.efield0001.cell.3d.hdf5'
dataNd_evec_comps=import_multdim_comps(filename=evecfilename)
dataNd_evec_with_outer_ghost_comps,num_ghost_evec=import_multdim_comps(filename=evecfilename,withghost=1)
title_var='E field'

fig=plot_Nd(dataNd_evec_comps,title=title_var)
mlab.savefig('fig_evec_mlab.png')
fig.scene.save_ps('fig_evec_mlab.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_evec_with_outer_ghost_comps,num_ghost_evec,title=title_var)
mlab.savefig('fig_evec_mlab_ghost.png')
fig.scene.save_ps('fig_evec_mlab_ghost.eps')
time.sleep(1)
mlab.close(all=True)

#############################
###############import density
densityfilename='./plt_density_plots/plt.1.hydrogen.density0001.3d.hdf5'
dataNd_density_comps=import_multdim_comps(filename=densityfilename)
dataNd_density_with_outer_ghost_comps,num_ghost_density=import_multdim_comps(filename=densityfilename,withghost=1)
title_var='density'

fig=plot_Nd(dataNd_density_comps,title=title_var)
mlab.savefig('fig_density_mlab_iso.png')
fig.scene.save_ps('fig_density_mlab_iso.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_density_with_outer_ghost_comps,num_ghost_density,title=title_var)
mlab.savefig('fig_density_mlab_ghost_iso.png')
fig.scene.save_ps('fig_density_mlab_ghost_iso.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_density_comps,title=title_var,sliced=1,x_slice=0.5,y_slice=0.5,z_slice=0.5)
mlab.savefig('fig_density_mlab_slice.png')
fig.scene.save_ps('fig_density_mlab_slice.eps')
time.sleep(1)
mlab.close(all=True)
fig=plot_Nd(dataNd_density_with_outer_ghost_comps,num_ghost_density,title=title_var,x_slice=0.5,y_slice=0.5,z_slice=0.5)
mlab.savefig('fig_density_mlab_ghost_slice.png')
fig.scene.save_ps('fig_density_mlab_ghost_slice.eps')
time.sleep(1)
mlab.close(all=True)


#collect figures
import Image
init_plotting('2x3')
f = pl.figure()
for n, fname in enumerate(('./fig_density_mlab_iso.png','./fig_density_mlab_slice.png','./fig_f_sum_mlab_slice.png','./fig_vpar_mu.png','./fig_vpar_mu_interp.png','./fig_vpar_compare.png')):
     image=Image.open(fname)#.convert("L")
     arr=np.asarray(image)
     ax=f.add_subplot(2, 3, n+1)
     ax.axis('off')
     pl.imshow(arr)
pl.tight_layout()
pl.savefig('fig0.png')
pl.show()





