import numpy as np
import matplotlib.pyplot as plt

from setupplot import init_plotting


import os, fnmatch
import ConfigParser


       
##############################################33
####################################################

############ speed up
init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)

os.environ['targetdir']='./scalingtest_singlenode_x'
from read_scale_dir import *
import read_scale_dir
speedup_23=1.0/pr_array_23WallTime/(1.0/pr_array_23WallTime[0])*pr_array_numTotalDecomp[0]
p0=plt.scatter(pr_array_numTotalDecomp,speedup_23,marker='o',linewidth=1,label='COGENT x')
os.environ['targetdir']='./scalingtest_singlenode_y'
reload(read_scale_dir)
from read_scale_dir import *
speedup_23=1.0/pr_array_23WallTime/(1.0/pr_array_23WallTime[0])*pr_array_numTotalDecomp[0]
p1=plt.scatter(pr_array_numTotalDecomp,speedup_23,marker='^',linewidth=1,label='COGENT y')
os.environ['targetdir']='./scalingtest_singlenode_z'
reload(read_scale_dir)
from read_scale_dir import *
speedup_23=1.0/pr_array_23WallTime/(1.0/pr_array_23WallTime[0])*pr_array_numTotalDecomp[0]
p2=plt.scatter(pr_array_numTotalDecomp,speedup_23,marker='x',linewidth=1,label='COGENT z')
os.environ['targetdir']='./scalingtest_singlenode_vpar'
reload(read_scale_dir)
from read_scale_dir import *
speedup_23=1.0/pr_array_23WallTime/(1.0/pr_array_23WallTime[0])*pr_array_numTotalDecomp[0]
p3=plt.scatter(pr_array_numTotalDecomp,speedup_23,marker='*',linewidth=1,label='COGENT vpar')
os.environ['targetdir']='./scalingtest_singlenode_mu'
reload(read_scale_dir)
from read_scale_dir import *
speedup_23=1.0/pr_array_23WallTime/(1.0/pr_array_23WallTime[0])*pr_array_numTotalDecomp[0]
p4=plt.scatter(pr_array_numTotalDecomp,speedup_23,marker='v',linewidth=1,label='COGENT mu')

plt.gca().set_xscale('log',basex=10)
plt.gca().set_yscale('log',basey=10)


alpha=1.0
N_8=pr_array_numTotalDecomp[0]**(1-alpha)
N=np.linspace(pr_array_numTotalDecomp[0],pr_array_numTotalDecomp[-1], 10, endpoint=True)
S=N_8*(N**(alpha))
p1,=plt.plot(N,S,':')#,label=r'$S =N_0^{1-\alpha} N^\alpha$')    
plt.gca().text(N[2]*1.0,S[2]*2.0,r'$\alpha=1.0$',color='b')
#plt.gca().annotate(r'$\alpha=-1.0$',xy=(N[2],Tw[2]),xytext=(N[2]*1.05,Tw[2]*1.05),arrowprops=dict(facecolor='none',shrink=0.02,width=0.1,headwidth=0.2))

alpha=0.5
N_8=pr_array_numTotalDecomp[0]**(1-alpha)
N=np.linspace(pr_array_numTotalDecomp[0],pr_array_numTotalDecomp[-1], 10, endpoint=True)
S=N_8*(N**(alpha))
p2,=plt.plot(N,S,'--')#,label=r'$S =N_0^{1-\alpha} N^\alpha$')    
plt.gca().text(N[2]*1.0,S[2]*0.75,r'$\alpha=0.5$',color='g')
#plt.gca().annotate(r'$T_w =C N^\alpha$',xy=(N[2],Tw[2]),xytext=(N[2]*1.15,Tw[2]*1.15),arrowprops=dict(facecolor='green',shrink=0.02,width=0.1,headwidth=0.2))

alpha=0.65
N_8=pr_array_numTotalDecomp[0]**(1-alpha)
N=np.linspace(pr_array_numTotalDecomp[0],pr_array_numTotalDecomp[-1], 10, endpoint=True)
S=N_8*(N**(alpha))
p3,=plt.plot(N,S,'-')#,label=r'$S =N_0^{1-\alpha} N^\alpha$')    
plt.gca().text(N[2]*1.0,S[2]*2.0,r'$\alpha=0.65$',color='r')
#plt.gca().annotate(r'$\alpha=-0.65$',xy=(N[2],Tw[2]),xytext=(N[2]*1.05,Tw[2]*1.05),arrowprops=dict(facecolor='none',shrink=0.02,width=0.1,headwidth=0.2))


plt.xlabel(r'$N$'+u' (no. of procs)')
plt.ylabel(r'$S$'+u' (Speedup Factors)\n'+r'$S=N_0 \Delta T_{w0}/\Delta T_w$')
plt.gca().legend(loc='best')
#plt.gca().legend([(p1,p2,p3),p0],[r'$T_w =C N^\alpha$','COGENT'],loc='best')

plt.tight_layout()
plt.savefig('speedup23.png')
plt.savefig('speedup23.eps')
plt.close('all')
#plt.clf()
#plt.show()




