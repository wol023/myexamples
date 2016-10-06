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
x_pt = 6
y_pt = 16
z_pt = 8
#############################################################
#############################################################
from plot_cogent_pack import *


with open('perun_cluster_target.txt','r') as fo:
    lines= fo.read().splitlines()
host=lines[0]
user=lines[1]
pword=lines[2]
basepath=lines[3]
targetpath=lines[4]
nextline=lines[5]
rawpathfiles=lines[6:]
print 'host=',host
print 'user=',user
print 'pword=',pword
print 'basepath=',basepath
print 'targetpath=',targetpath
print nextline
#print rawpathfiles

pathfiles=[]
for i in range(len(rawpathfiles)):
    if rawpathfiles[i].lstrip().startswith('#'): #skip comment
        continue
    line =rawpathfiles[i].rstrip() #skip blank line
    if not line:
        continue 
    else:
        pathfiles.append(line)

print pathfiles


paths=[]
files=[]
for i in range(len(pathfiles)):
    head=os.path.split(pathfiles[i])
    paths.append(head[0])
    files.append(head[1])
print paths
print files


plot_output='./python_auto_plots.0.dir'
notunique=1
while notunique:
    if os.path.exists(plot_output):
        notunique=1
        int_oldnum=int(plot_output.split('./python_auto_plots.')[1].split('.dir')[0])
        str_newnum=str(int_oldnum+1)
        plot_output='./python_auto_plots.'+str_newnum+'.dir'
    else:
        notunique=0


for i in range(len(pathfiles)):
    status=check_and_try_cluster(pathfiles[i],host=host,username=user,password=pword,basepath=basepath,targetpath=targetpath)
    if status>0:
        if 'dfn' in pathfiles[i]:
            if 'hydrogen' in pathfiles[i]:
                speciesname='i'
            elif 'electron' in pathfiles[i]:
                speciesname='e'
            else:
                speciesname='s'
            plot_dfn(pathfiles[i],speciesname=speciesname,x_pt=x_pt,y_pt=y_pt,z_pt=z_pt,targetdir=plot_output)
        if 'potential' in pathfiles[i]:
            plot_potential(pathfiles[i],ghost=0,x_slice=0.5,y_slice=0.5,z_slice=0.5,targetdir=plot_output)
        if 'BField' in pathfiles[i]:
            plot_bvec(pathfiles[i],ghost=0,targetdir=plot_output)
        if 'efield' in pathfiles[i]:
            plot_evec(pathfiles[i],ghost=0,targetdir=plot_output)
        if 'density' in pathfiles[i]:
            if 'hydrogen' in pathfiles[i]:
                speciesname='i'
            elif 'electron' in pathfiles[i]:
                speciesname='e'
            else:
                speciesname='s'
            plot_density(pathfiles[i],ghost=0,speciesname=speciesname,targetdir=plot_output)
    else:
        print 'No', pathfiles[i], 'is found.. skipping the file.'

################collect figures

figure_list=os.listdir(plot_output)
demo_list_full=[]
for i in range(len(figure_list)):
    if 'png' in figure_list[i]:
        demo_list_full.append(plot_output+'/'+figure_list[i]);

print demo_list_full

import fnmatch

demo_list=[d_list for d_list in demo_list_full if not fnmatch.fnmatch(d_list,'*/fig[0-9].png')]
demo_list.sort()


canverse_num = len(demo_list)/6
if len(demo_list)%6:
    canverse_num+=1

import Image

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
    demo_sum=plot_output+'/fig'+'%d'%num_can+'.png'
    pl.savefig(demo_sum)
    print demo_sum, ' written.'
pl.show()

# bash command to make animated gif
# convert -delay 50 -loop 0 *.png (name).gif




