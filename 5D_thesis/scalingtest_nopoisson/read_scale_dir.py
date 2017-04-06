import os, fnmatch
import numpy as np
from pprint import pprint
targetdir=os.environ['targetdir']

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result

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
 

dirs = [d for d in os.listdir(targetdir) if os.path.isdir(os.path.join(targetdir, d))]
#dirs = [d for d in os.listdir('./') if os.path.isdir(os.path.join('./', d))]
print dirs
fname=[]
for d,dd in enumerate(dirs):
    dirs[d]=targetdir+'/'+dirs[d]
print dirs

for d in dirs:
    print d,type(d)
    fname_each=find('slurm-*.out', d)
    if len(fname_each)>0:
        fname.append(fname_each[0])
print fname

if len(fname)==0:
	fname=find('perunoutput.out', './')

#
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



