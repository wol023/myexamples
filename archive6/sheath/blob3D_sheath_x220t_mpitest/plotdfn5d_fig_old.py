import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1 
from scipy import interpolate
from scipy import ndimage
import pylab as pl
from scipy.optimize import leastsq

from mayavi import mlab


#setup plot
graphDPI =300
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


# import dfn
File =h5py.File('./plt_dfn_plots/plt.1.hydrogen.dfn0000.5d.hdf5','r')     

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
print shifter

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
print num_cell_total
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
print num_cell_loc
prod_num_cell_loc = np.prod(num_cell_loc)
print 'prod_num_cell_loc=',prod_num_cell_loc
     
data = File['level_0']['data:datatype=0'][:]
previous_index=0

if dim<5:
    dataNd=np.linspace(0.0,0.1,num=prod_num_cell_total).reshape((num_cell_total[0],num_cell_total[1],num_cell_total[2],num_cell_total[3]) )
else:
    dataNd=np.linspace(0.0,0.1,num=prod_num_cell_total).reshape((num_cell_total))
num_cell_loc=num_cell_loc.astype(int)

for i in range(num_decomposition):
    cells = File['level_0']['boxes'][i]
    print 'decomposition=',i
    print 'cells=',cells
    
    cells_shift=cells
    for j in range(len(cells_shift)):
        cells_shift[j]=cells[j]+shifter[j]
    
        
    if dim<5:
        dataNd_loc=data[previous_index:prod_num_cell_loc*(i+1)].reshape((num_cell_loc[dir_x],num_cell_loc[dir_y],num_cell_loc[dir_vpar],num_cell_loc[dir_mu]),order='F')
    else:
        print previous_index
        print prod_num_cell_loc*(i+1)
        print prod_num_cell_loc
        print prod_num_cell_total
        print (prod_num_cell_loc*(i+1)-previous_index)
        print prod_num_cell_total/(prod_num_cell_loc*(i+1)-previous_index)

        dataNd_loc=data[previous_index:prod_num_cell_loc*(i+1)].reshape((num_cell_loc[dir_x],num_cell_loc[dir_y],num_cell_loc[dir_z],num_cell_loc[dir_vpar],num_cell_loc[dir_mu]),order='F')

    previous_index=prod_num_cell_loc*(i+1)

    print cells_shift
    print cells_shift[dir_x],cells_shift[dim+dir_x]+1
    print cells_shift[dir_y],cells_shift[dim+dir_y]+1
    print cells_shift[dir_z],cells_shift[dim+dir_z]+1
    print cells_shift[dir_vpar],cells_shift[dim+dir_vpar]+1
    print cells_shift[dir_mu],cells_shift[dim+dir_mu]+1



    dataNd[cells_shift[dir_x]:cells_shift[dim+dir_x]+1, cells_shift[dir_y]:cells_shift[dim+dir_y]+1, cells_shift[dir_z]:cells_shift[dim+dir_z]+1, cells_shift[dir_vpar]:cells_shift[dim+dir_vpar]+1, cells_shift[dir_mu]:cells_shift[dim+dir_mu]+1]=dataNd_loc


File.close()
print dataNd.shape

num_xcell=num_cell_total[dir_x] #
num_ycell=num_cell_total[dir_y] #
if dim>=5:
    num_zcell=num_cell_total[dir_z] #
num_vparcell=num_cell_total[dir_vpar] #
num_mucell=num_cell_total[dir_mu] #

# diagnostic points
x_pt = 4
y_pt = 5
z_pt = 6
vpar_rbf_refine = 5
mu_rbf_refine = 5
rbf_smooth = 0

#phase
VPAR,MU = np.mgrid[-1:1:(num_vparcell*1j),0:1:(num_mucell*1j)]

#refine
if dim<5:
    rbf = interpolate.Rbf(VPAR.ravel(), MU.ravel(), dataNd[x_pt,y_pt,:,:].ravel(), smooth=rbf_smooth)
else:
    rbf = interpolate.Rbf(VPAR.ravel(), MU.ravel(), dataNd[x_pt,y_pt,z_pt,:,:].ravel(), smooth=rbf_smooth)
    
VPAR_REFINE, MU_REFINE = np.mgrid[-1:1:(num_vparcell*1j*vpar_rbf_refine), 0:1:(num_mucell*1j*mu_rbf_refine)]
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

if dim<5:
    delta_f_model = (f_model-dataNd[x_pt,y_pt,:,:])
else:
    delta_f_model = (f_model-dataNd[x_pt,y_pt,z_pt,:,:])

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

f_mu_sum_rbf = np.zeros(len(VPAR_REFINE[:,0]))
for i in range(len(VPAR_REFINE[:,0])):
    for j in range(len(VPAR_REFINE[0,:])):
        f_mu_sum_rbf[i]=f_mu_sum_rbf[i]+saved_rbf[i,j]
    f_mu_sum_rbf[i]=f_mu_sum_rbf[i]/len(VPAR_REFINE[0,:])



#raw density 2D 
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    im=plt.contourf(X,Y,f_vpar_mu_sum,100)
else:
    im=plt.contourf(X[:,:,0],Y[:,:,0],f_vpar_mu_sum[:,:,z_pt],100)
plt.title(u'n(X,Y)')
plt.xlabel(u'X')
plt.ylabel(u'Y')
plt.colorbar()
plt.tight_layout()
plt.savefig('fig1.png')
plt.savefig('fig1.eps')
plt.close('all')

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


#raw f
init_plotting()
plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    plt.contourf(VPAR,MU,dataNd[x_pt,y_pt,:,:],100)
else:
    plt.contourf(VPAR,MU,dataNd[x_pt,y_pt,z_pt,:,:],100)
plt.title(r'$f(\bar{v}_\parallel,\bar{\mu})$')
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(r'$\bar{\mu}$')
plt.colorbar();
plt.tight_layout()
plt.savefig('fig2.png')
plt.savefig('fig2.eps')
plt.close('all')

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
plt.savefig('fig2_rbf.png')
plt.savefig('fig2_rbf.eps')
plt.close('all')

##reconstructed model maxwellian
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
plt.savefig('fig2_model.png')
plt.savefig('fig2_model.eps')
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
plt.savefig('fig2_delta_model.png')
plt.savefig('fig2_delta_model.eps')
plt.close('all')


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

#summed maxwellian on refined 
init_plotting()
fig=plt.subplot(111)
plt.plot(VPAR_REFINE[:,0], f_mu_sum_rbf )
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel('<f>')
plt.tight_layout()
plt.savefig('fig3_rbf.png')
plt.savefig('fig3_rbf.eps')
plt.close('all')

#compare mu slice with model
init_plotting()
fig=plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
if dim<5:
    plt.plot(VPAR[:,0], f_model[:,0]/dataNd[x_pt,y_pt,:,0])
else:
    plt.plot(VPAR[:,0], f_model[:,0]/dataNd[x_pt,y_pt,z_pt,:,0])
plt.xlabel(r'$\bar{v}_\parallel$')
plt.ylabel(u'f_model/f(MU=0)')
plt.tight_layout()
plt.savefig('fig4_mu0_comp_model.png')
plt.savefig('fig4_mu0_comp_model.eps')
plt.close('all')

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
from PIL import Image
init_plotting('2x3')
f = pl.figure()
for n, fname in enumerate(('fig1_mlab_iso.png','fig1_mlab_slice.png','fig2_im.png','fig2.png','fig3.png','fig4_mu0_comp_fit.png')):
     image=Image.open(fname)#.convert("L")
     arr=np.asarray(image)
     ax=f.add_subplot(2, 3, n+1)
     ax.axis('off')
     pl.imshow(arr)
pl.tight_layout()
pl.savefig('fig0.png')
pl.show()





