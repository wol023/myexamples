import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import sys

import SlopeFinder as SlopeFinder

def init_plotting():
    plt.rcParams['figure.figsize'] = (5, 4)
    plt.rcParams['font.size'] = 10
#    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    #plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['axes.linewidth'] = 1.2

#    plt.gca().spines['right'].set_color('none')
#    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')


mycolors=('k','y','m','c','b','g','r','#aaaaaa')
mylinestyles=('-','--','-.',':')
mystyles=[(mycolor,mylinestyle) for mylinestyle in mylinestyles for mycolor in mycolors]




if len(sys.argv)==1:
    filename = 'npp.dat'
else:
    filename = str(sys.argv[1])

print filename

row_size = sum(1 for line in open(filename))-1
col_size=1
headlines=[]

with open(filename, 'r') as f:
    for line in f:
        if line.lstrip().startswith('VARIABLES'): 
            line =line.rstrip() 
            strippedline=line.lstrip('VARIABLES="')
            strippedline=strippedline.rstrip('"')
            headlines = strippedline.split('","')
            for head in headlines:
                print head
                
            print headlines
            col_size=len(headlines)
            break
        else:
            if '\t' in line:
                headlines = line.split("\t")
            else:
                headlines= line.split(" ")
            print headlines

            col_size=len(headlines)
            break

print row_size, col_size

X=np.zeros((row_size,1))
Y=np.zeros((row_size,col_size-1))
print X.shape
print Y.shape


l = 0  #line count
with open(filename, 'r') as f:
    for line in f:
        if line.lstrip()[0].isalpha(): 
            continue
        if not line:
            continue 
        else: #non head line
            strippedline=line.rstrip('\r\n')
            strippedline=strippedline.rstrip('\n')
            if '\t' in strippedline:
                lhsrhs = strippedline.split("\t")
            else:
                lhsrhs = strippedline.split(" ")
            c=0
            #print lhsrhs
            while c<len(lhsrhs): 
                #print 'c=',c
                if c==0:
                    X[l]=float(lhsrhs[c])
                    c=c+1
                else:
                    Y[l,c-1]=float(lhsrhs[c])
                    c=c+1
            l = l+1


init_plotting()
fig = plt.subplot(111)
ax = plt.gca()
ax.margins(0., 0.)

i=0
while i<col_size-1:
    lines=ax.plot((X),(Y[:,i]),label=headlines[i+1],color=mystyles[i][0],linestyle=mystyles[i][1])
    i=i+1

af =  SlopeFinder.SlopeFinder(ax=ax)
#af =  SlopeFinder.SlopeFinder(X,Y[:,i-1], ax=ax)
plt.connect('button_press_event', af)
plt.connect('key_press_event', af.on_key_press)
plt.connect('key_release_event', af.on_key_release)

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel(r'$k$')
plt.ylabel(r'$\hat{\pi}_{\pi}$')

plt.savefig(filename.replace('.dat','.eps'))

plt.show()

