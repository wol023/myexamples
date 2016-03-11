import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import os, fnmatch
import ConfigParser



def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result


print 'current directory:',os.getcwd()

te_list = []
ti_list = []
omega_star_point_list = []
omega_star_fit_omega_star_point_list_te = []

dirlist = ['expobol_64x128x16x16_dt005_new_vlasov_20_te2_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_20_te3_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_20_te4_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_20_te5_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_20_te7_5_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_20_te10_diri'
        ]

for count, item in enumerate(dirlist):
    os.chdir(item)
    
    print 'current working directory:',os.getcwd()
    
    
    import subprocess
    subprocess.call("python readinginput.py", shell=True)
    #read finish.txt file
    with open('finish.txt', 'r') as f:
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
                if 'te' in lhsrhs[0]:
                    te_list.append(float(lhsrhs[1]))
                if 'ti' in lhsrhs[0]:
                    ti_list.append(float(lhsrhs[1]))
                if 'omega_star_point' in lhsrhs[0]:
                    omega_star_point_list.append(float(lhsrhs[1]))
                if 'omega_star_fit/omega*_point' in lhsrhs[0]:
                    omega_star_fit_omega_star_point_list_te.append(float(lhsrhs[1]))
    os.chdir('../')



####



te_list = []
ti_list = []
omega_star_point_list = []
omega_star_fit_omega_star_point_list_teti = []

dirlist = ['expobol_64x128x16x16_dt005_new_vlasov_teti40_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_teti60_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_teti80_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_teti100_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_teti150_diri',
        'expobol_64x128x16x16_dt005_new_vlasov_teti200_diri'
        ]

for count, item in enumerate(dirlist):
    os.chdir(item)
    
    print 'current working directory:',os.getcwd()
    
    
    import subprocess
    subprocess.call("python readinginput.py", shell=True)
    #read finish.txt file
    with open('finish.txt', 'r') as f:
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
                if 'te' in lhsrhs[0]:
                    te_list.append(float(lhsrhs[1]))
                if 'ti' in lhsrhs[0]:
                    ti_list.append(float(lhsrhs[1]))
                if 'omega_star_point' in lhsrhs[0]:
                    omega_star_point_list.append(float(lhsrhs[1]))
                if 'omega_star_fit/omega*_point' in lhsrhs[0]:
                    omega_star_fit_omega_star_point_list_teti.append(float(lhsrhs[1]))
    os.chdir('../')



# set global settings
def init_plotting():
    plt.rcParams['figure.figsize'] = (4, 3)
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
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = 1

#    plt.gca().spines['right'].set_color('none')
#    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

init_plotting()

#x = [0.31415*xi for xi in xrange(0,10)]
#y1 = [sin(xi) for xi in x]
#y2 = [cos(xi + 0.5) for xi in x]
#y3 = [cos(xi + 0.5) + sin(xi) for xi in x]

# begin subplots region
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
#plt.plot(x, y1, linestyle='-', marker='.', linewidth=1, color='r', label='sin')
#plt.plot(x, y2, linestyle='.', marker='o', linewidth=1, color='b', label='cos')

plt.plot(np.array(te_list),np.array(omega_star_fit_omega_star_point_list_te),linestyle ='-', marker='x', linewidth=1, color='r',label='Ti=20 eV')
plt.plot(np.array(te_list),np.array(omega_star_fit_omega_star_point_list_teti),linestyle='--', marker='o',linewidth=1, color='b', label='Ti=Te')

#plt.gca().annotate(u'point $\\frac{\\tau}{2}$', xy=(x[2], y1[2]),  xycoords='data',
#                xytext=(30, -10), textcoords='offset points', size=8,
#                arrowprops=dict(arrowstyle='simple', fc='g', ec='none'))

plt.xlabel(u'Te (eV)')
plt.ylabel(u'$\omega / \omega_*$')
plt.title(u'Drift wave frequency')

plt.gca().legend(bbox_to_anchor = (0.0, 0.1))

#plt.subplot(122)
#plt.gca().margins(0.1, 0.1)
#plt.plot(x, y3, linestyle='--', marker='.', linewidth=1, color='g', label='sum')
#
#plt.gca().annotate(u'$y_x$', xy=(x[2], y3[2]),  xycoords='data',
#                xytext=(-30, -20), textcoords='offset points', size=8,
#                arrowprops=dict(arrowstyle='simple', fc='orange', ec='none'))
#
#plt.xlabel(u'x label')
#plt.ylabel(u'y label')
#plt.title(u'Second plot title')
#
#plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
# end subplots region

# output resulting plot to file
plt.ylim(0.8,1.0)

plt.tight_layout()
plt.savefig('graph.png')
plt.savefig('graph.eps')








