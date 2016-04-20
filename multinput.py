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
omega_star_fit_omega_star_1_chi2_list_te = []

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
                if 'omega_star_fit/omega*_1_chi2' in lhsrhs[0]:
                    omega_star_fit_omega_star_1_chi2_list_te.append(float(lhsrhs[1]))
    os.chdir('../')



####



te_list = []
ti_list = []
omega_star_point_list = []
omega_star_fit_omega_star_point_list_teti = []
omega_star_fit_omega_star_1_chi2_list_teti = []

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
                if 'omega_star_fit/omega*_1_chi2' in lhsrhs[0]:
                    omega_star_fit_omega_star_1_chi2_list_teti.append(float(lhsrhs[1]))
    os.chdir('../')


from setupplot import init_plotting

#### first plot

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
plt.ylabel(r'$\omega_{\mathrm{fit}} / \omega_*$',fontsize=1.5*plt.rcParams['font.size'])
#plt.title(u'Drift wave frequency')

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
plt.ylim(0.8,1.05)

plt.tight_layout()
plt.savefig('graph1.png')
plt.savefig('graph1.eps')


plt.close()
####second plot


init_plotting()

# begin subplots region
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
#plt.plot(x, y1, linestyle='-', marker='.', linewidth=1, color='r', label='sin')
#plt.plot(x, y2, linestyle='.', marker='o', linewidth=1, color='b', label='cos')

plt.plot(np.array(te_list),np.array(omega_star_fit_omega_star_1_chi2_list_te),linestyle ='-', marker='x', linewidth=1, color='r',label='Ti=20 eV')
plt.plot(np.array(te_list),np.array(omega_star_fit_omega_star_1_chi2_list_teti),linestyle='--', marker='o',linewidth=1, color='b', label='Ti=Te')

#plt.gca().annotate(u'point $\\frac{\\tau}{2}$', xy=(x[2], y1[2]),  xycoords='data',
#                xytext=(30, -10), textcoords='offset points', size=8,
#                arrowprops=dict(arrowstyle='simple', fc='g', ec='none'))

plt.xlabel(u'Te (eV)')
plt.ylabel(r'$\omega_{\mathrm{fit}} / (\omega_*/(1+k_y^2\rho_s^2)) $',fontsize=1.5*plt.rcParams['font.size'])
#plt.title(u'Drift wave frequency')

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
plt.ylim(0.8,1.05)

plt.tight_layout()
plt.savefig('graph2.png')
plt.savefig('graph2.eps')



