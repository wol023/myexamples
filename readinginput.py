import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import os, fnmatch
import ConfigParser

#read output file

ref_time=0.0

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result

fname=find('slurm-*.out', './')

with open(fname[0], 'r') as f:
    for line in f:
        if line.lstrip().startswith('*'): #skip comment
            continue
        line =line.rstrip() #skip blank line
        if not line:
            continue 
        else: #noncomment line
            strippedline=line
            lhsrhs = strippedline.split(":")
            l=0
            while l<len(lhsrhs): #strip white spaces in lhs
                lhsrhs[l]=lhsrhs[l].rstrip()
                lhsrhs[l]=lhsrhs[l].lstrip()
                l=l+1
            #print type( lhsrhs[0])
            if 'TRANSIT' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                ref_time=float(lhsrhs[1])
f.closed
print 'ref_time = ',ref_time

#read input file
fname=find('*.in', './')

with open(fname[0], 'r') as f:
    for line in f:
        if line.lstrip().startswith('#'): #skip comment
            continue
        line =line.rstrip() #skip blank line
        if not line:
            continue 
        else: #noncomment line
            strippedline=line
            lhsrhs = strippedline.split("=")
            l=0
            while l<len(lhsrhs): #strip white spaces in lhs
                lhsrhs[l]=lhsrhs[l].rstrip()
                lhsrhs[l]=lhsrhs[l].lstrip()
                l=l+1
            #print type( lhsrhs[0])
            if 'N0_0_grid_func.function' in lhsrhs[0]:
                print lhsrhs[0],'=',lhsrhs[1]
                n0_gridfunc=lhsrhs[1][1:-1]
f.closed
print 'n0_gridfunc = ',n0_gridfunc

####

from sympy.parsing.sympy_parser import parse_expr
#from sympy.parsing.sympy_parser import standard_transformations
#from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy import *
from sympy.abc import x, y, z
from sympy.utilities.lambdify import implemented_function
from sympy import Function


print type(n0_gridfunc)
pe=parse_expr(n0_gridfunc)
print type(pe)
print pe

f = lambdify(x, pe)
print f(pi)

xx = np.linspace(0.0, np.pi*2, 100)
yy = np.linspace(0.0, np.pi*2, 100)

for i in range(len(xx)):
    yy[i] = f(xx[i])

dx = 1.0/len(xx)
dyydx = np.gradient(yy)/dx

print min(dyydx)

fig, ax = plt.subplots(2,1)
ax[0].plot(xx,yy )
ax[1].plot(xx,dyydx )
plt.show()


