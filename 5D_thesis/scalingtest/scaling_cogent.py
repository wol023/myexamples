import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from setupplot import init_plotting


import os, fnmatch
import ConfigParser


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result

dirs = [d for d in os.listdir('./') if os.path.isdir(os.path.join('./', d))]
print dirs
fname=[]
for d in dirs:
    print d,type(d)
    fname_each=find('slurm-*.out', d)
    if len(fname_each)>0:
        fname.append(fname_each[0])
print fname

if len(fname)==0:
	fname=find('perunoutput.out', './')

print '************ SLURM OUT FILE *****************'
with open('finish.txt', 'wb') as fh:
    buf = "************ SLURM OUT FILE *****************\n"
    fh.write(buf)

class RunCase:
    def __init__(self,name):
        self.name=name
        self.decompConf_filled=0
        self.decompVel_filled=0
        self.numTotalDecomp=1
        self.numDecompConf=[]
        self.numDecompVel=[]
        self.numStep=0
        self.finalStep=0
        self.finalSolveWallTime=0.0
        self.finalTotalWallTime=0.0
        self.simTime=[]
        self.wallTime=[]
        self.strDecompConf=""
        self.strDecompVel=""
    def set_numTotalDecomp(self,numTotalDecomp):
        self.numTotalDecomp=numTotalDecomp
    def add_numDecompConf(self,num):
        if self.decompConf_filled==0:
            self.numDecompConf.append(num)
            self.numTotalDecomp*=num
    def add_numDecompVel(self,num):
        if self.decompVel_filled==0:
            self.numDecompVel.append(num)
            self.numTotalDecomp*=num
    def fix_decompConf(self):
        self.decompConf_filled=1
        for dv in self.numDecompConf:
            self.strDecompConf=self.strDecompConf+str(dv)+str('x')
        self.strDecompConf=self.strDecompConf[:-1]
    def fix_decompVel(self):
        self.decompVel_filled=1
        for dv in self.numDecompVel:
            self.strDecompVel=self.strDecompVel+str(dv)+str('x')
        self.strDecompVel=self.strDecompVel[:-1]
    def add_numStep(self):
        self.numStep+=1
    def add_simTime(self,time):
        self.simTime.append(time)
    def add_wallTime(self,time):
        self.wallTime.append(time)
    def set_finalSteps(self,steps):
        self.finalStep=steps
    def set_finalSolveWallTime(self,finalswt):
        self.finalSolveWallTime=finalswt
    def set_finalTotalWallTime(self,finaltwt):
        self.finalTotalWallTime=finaltwt
        

runs=[]
dummy = [0]
#for l in dummy:
for l,ll in enumerate(fname):
    with open(fname[l], 'r') as f:
        runs.append(RunCase(fname[l]))
        print runs[l].name
        for line in f:
            if line.lstrip().startswith('#'): #skip comment
                continue
            line =line.rstrip() #skip blank line
            if not line:
                continue 
            else: #noncomment line
                strippedline=line
                lhsrhs = strippedline.split("=")
                ld=0
                while ld<len(lhsrhs): #strip white spaces in lhs
                    lhsrhs[ld]=lhsrhs[ld].rstrip()
                    lhsrhs[ld]=lhsrhs[ld].lstrip()
                    ld=ld+1
                if len(lhsrhs)==2:
                    if 'configuration_decomposition' in lhsrhs[0]:
                        if runs[l].decompConf_filled==0:
                            print lhsrhs[0],'=',lhsrhs[1]
                            rhs_each=lhsrhs[1].split(" ")
                            for item in rhs_each:
                                runs[l].add_numDecompConf(int(item))
                            runs[l].fix_decompConf()
                    if 'velocity_decomposition' in lhsrhs[0]:
                        if runs[l].decompVel_filled==0:
                            print lhsrhs[0],'=',lhsrhs[1]
                            rhs_each=lhsrhs[1].split(" ")
                            for item in rhs_each:
                                runs[l].add_numDecompVel(int(item))
                            runs[l].fix_decompVel()
                else:
                    strippedline=line
                    lhsrhs = strippedline.split(", solver wall time is ")
                    if len(lhsrhs)==2:
                        if runs[l].numStep<3: # number of max time step
                            #print lhsrhs
                            lhs=lhsrhs[0]
                            rhs=lhsrhs[1]
                            llhs=lhs.split(" completed, simulation time is ")
                            tempstep=int(llhs[0].lstrip("Step "))
                            tempsimtime=float(llhs[1])
                            tempwalltime=float(rhs.rstrip(" seconds"))
                            runs[l].add_numStep()
                            runs[l].add_simTime(tempsimtime)
                            runs[l].add_wallTime(tempwalltime)
                    else:
                        lhsrhs = strippedline.split("    Time steps: ")
                        if len(lhsrhs)==2:
                            runs[l].set_finalSteps(int(lhsrhs[1]))
                        else:
                            lhsrhs = strippedline.split("Solve wall time (in seconds): ")
                            if len(lhsrhs)==2:
                                runs[l].set_finalSolveWallTime(float(lhsrhs[1]))
                            else:
                                lhsrhs = strippedline.split("Total wall time (in seconds): ")
                                if len(lhsrhs)==2:
                                    runs[l].set_finalTotalWallTime(float(lhsrhs[1]))




#legend_data_fit_with_growth = r'$\gamma/\omega^*$'+' = %g'% ( lin_fitted_logy2[0]/2.0/omega_star_analytic)+'\n'+r'$\gamma/\omega^*_d$'+' = %g'% ( lin_fitted_logy2[0]/2.0/omega_star_analytic*(1.0+chi*chi))

print 
print 'runs'
for l,ll in enumerate(runs):
    print runs[l].numStep, runs[l].numTotalDecomp, runs[l].finalTotalWallTime

#get only completed runs
complete_runs = []
for l,ll in enumerate(runs):
    if runs[l].numStep>=3:
        complete_runs.append(runs[l])

print
print 'clean'
print
print 'complete_runs'
for l,ll in enumerate(complete_runs):
    print complete_runs[l].numStep, complete_runs[l].numTotalDecomp, complete_runs[l].finalTotalWallTime

print 
print 'sort'
def getTotalDecomp(run):
    return run.numTotalDecomp
sorted_runs=sorted(complete_runs, key=getTotalDecomp)


print
print 'sorted_runs'
for l,ll in enumerate(sorted_runs):
    print sorted_runs[l].numTotalDecomp, sorted_runs[l].finalTotalWallTime


pprint (vars(sorted_runs[1]))
pr_array_numTotalDecomp= np.array([])
pr_array_finalSolveWallTime= np.array([])
pr_array_finalTotalWallTime= np.array([])
pr_array_01WallTime= np.array([])
pr_array_12WallTime= np.array([])
pr_array_23WallTime= np.array([])
pr_array_3fWallTime= np.array([])
pr_array_decomp=[]


for l,ll in enumerate(sorted_runs):
    pr_array_numTotalDecomp = np.append(pr_array_numTotalDecomp, sorted_runs[l].numTotalDecomp)
    pr_array_finalSolveWallTime= np.append(pr_array_finalSolveWallTime, sorted_runs[l].finalSolveWallTime)
    pr_array_finalTotalWallTime= np.append(pr_array_finalTotalWallTime, sorted_runs[l].finalTotalWallTime)
    pr_array_01WallTime= np.append(pr_array_01WallTime, sorted_runs[l].wallTime[0])
    pr_array_12WallTime= np.append(pr_array_12WallTime, sorted_runs[l].wallTime[1]-sorted_runs[l].wallTime[0])
    pr_array_23WallTime= np.append(pr_array_23WallTime, sorted_runs[l].wallTime[2]-sorted_runs[l].wallTime[1])
    pr_array_3fWallTime= np.append(pr_array_3fWallTime, sorted_runs[l].finalTotalWallTime-sorted_runs[l].wallTime[2])
    pr_array_decomp.append(sorted_runs[l].strDecompConf+'x'+sorted_runs[l].strDecompVel)

print pr_array_numTotalDecomp 
print pr_array_finalSolveWallTime
print pr_array_finalTotalWallTime
print pr_array_01WallTime
print pr_array_12WallTime
print pr_array_23WallTime
print pr_array_3fWallTime

print pr_array_01WallTime+pr_array_12WallTime+pr_array_23WallTime+pr_array_3fWallTime
print pr_array_decomp


        
init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)

plt.plot(pr_array_numTotalDecomp,pr_array_01WallTime,marker='.',linewidth=1,label='0-1 step')
plt.plot(pr_array_numTotalDecomp,pr_array_01WallTime+pr_array_12WallTime,marker='.',linewidth=1,label='0-2 step')
plt.plot(pr_array_numTotalDecomp,pr_array_01WallTime+pr_array_12WallTime+pr_array_23WallTime,marker='.',linewidth=1,label='0-3 step')
plt.plot(pr_array_numTotalDecomp,pr_array_finalTotalWallTime,marker='.',linewidth=1,label='0-3-final')

for l in range(len(pr_array_decomp)):
    plt.gca().text(pr_array_numTotalDecomp[l],pr_array_finalTotalWallTime[l],pr_array_decomp[l])

plt.gca().set_xscale('log',basex=2)
#plt.gca().set_yscale('log',basey=2)
#plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
#plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlabel(u'no. of processes')
plt.ylabel(u'Wall Time (s)')
plt.gca().legend(loc='best')
plt.tight_layout()
plt.savefig('scaling.png')
plt.savefig('scaling.eps')
#plt.show()
plt.close('all')
plt.clf()


init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)

p0=plt.scatter(pr_array_numTotalDecomp,pr_array_23WallTime,marker='o',linewidth=1,label='COGENT')

plt.gca().set_xscale('log',basex=10)
plt.gca().set_yscale('log',basey=10)
#plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
#plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))

#for l in range(len(pr_array_decomp)):
    #plt.gca().text(pr_array_numTotalDecomp[l],pr_array_23WallTime[l],pr_array_decomp[l])
    #plt.gca().annotate(pr_array_decomp[l],xy=(pr_array_numTotalDecomp[l],pr_array_23WallTime[l]),xytext=(pr_array_numTotalDecomp[l]*1.1,pr_array_23WallTime[l]*1.1),arrowprops=dict(facecolor='black',shrink=0.02,width=0.1,headwidth=0.2))


alpha=-1.0
C=pr_array_23WallTime[0]/pr_array_numTotalDecomp[0]**alpha
N=np.linspace(pr_array_numTotalDecomp[0],pr_array_numTotalDecomp[-1], 10, endpoint=True)
Tw=C*(N**(alpha))
p1,=plt.plot(N,Tw,':',label=r'$T_w =C N^\alpha, \alpha=-1.0$')    
plt.gca().text(N[2]*1.05,Tw[2]*1.05,r'$\alpha=-1.0$')
#plt.gca().annotate(r'$\alpha=-1.0$',xy=(N[2],Tw[2]),xytext=(N[2]*1.05,Tw[2]*1.05),arrowprops=dict(facecolor='none',shrink=0.02,width=0.1,headwidth=0.2))

alpha=-0.5
C=pr_array_23WallTime[0]/pr_array_numTotalDecomp[0]**alpha
Tw=C*(N**(alpha))
p2,=plt.plot(N,Tw,'--',label=r'$T_w =C N^\alpha, \alpha=-0.5$')    
plt.gca().text(N[2]*1.05,Tw[2]*1.05,r'$\alpha=-0.5$')
#plt.gca().annotate(r'$T_w =C N^\alpha$',xy=(N[2],Tw[2]),xytext=(N[2]*1.15,Tw[2]*1.15),arrowprops=dict(facecolor='green',shrink=0.02,width=0.1,headwidth=0.2))

alpha=-0.65
C=pr_array_23WallTime[0]/pr_array_numTotalDecomp[0]**alpha
Tw=C*(N**(alpha))
p3,=plt.plot(N,Tw,'-',label=r'$T_w =C N^\alpha, \alpha = -0.65$')    
plt.gca().text(N[2]*1.05,Tw[2]*1.05,r'$\alpha=-0.65$')
#plt.gca().annotate(r'$\alpha=-0.65$',xy=(N[2],Tw[2]),xytext=(N[2]*1.05,Tw[2]*1.05),arrowprops=dict(facecolor='none',shrink=0.02,width=0.1,headwidth=0.2))


plt.xlabel(r'$N$'+u' (no. of processes)')
plt.ylabel(r'$T_w$'+u' (Wall Time) [s]')
plt.gca().legend(loc='best')
#plt.gca().legend([(p1,p2,p3),p0],[r'$T_w =C N^\alpha$','COGENT'],loc='best')

plt.tight_layout()
plt.savefig('scaling12.png')
plt.savefig('scaling12.eps')
#plt.close('all')
#plt.clf()
plt.show()




