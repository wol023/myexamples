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
        self.simTime=[]
        self.wallTime=[]
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
    def fix_decompVel(self):
        self.decompVel_filled=1
    def add_numStep(self):
        self.numStep+=1
    def add_simTime(self,time):
        self.simTime.append(time)
    def add_wallTime(self,time):
        self.wallTime.append(time)
        

runs=[]
for l,ll in enumerate(fname):
    print l,ll
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
                            print lhsrhs
                            lhs=lhsrhs[0]
                            rhs=lhsrhs[1]
                            llhs=lhs.split(" completed, simulation time is ")
                            tempstep=int(llhs[0].lstrip("Step "))
                            tempsimtime=float(llhs[1])
                            tempwalltime=float(rhs.rstrip(" seconds"))
    
                            runs[l].add_numStep()
                            runs[l].add_simTime(tempsimtime)
                            runs[l].add_wallTime(tempwalltime)
                            
    pprint (vars(runs[l]))




 


