import matplotlib.pyplot as plt
graphDPI =200

# set global settings
def init_plotting(form=''):
    if (form == '2x3'):
        plt.rcParams['figure.figsize'] = (16, 9)
#        plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    elif (form == '2x2'):
        plt.rcParams['figure.figsize'] = (12, 9)
#        plt.rcParams['font.size'] = 10
#    plt.rcParams['font.family'] = 'Times New Roman'
#        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
#        plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
#        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
#        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
#        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
#        plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
#        plt.rcParams['xtick.major.size'] = 0
#        plt.rcParams['xtick.minor.size'] = 0
#        plt.rcParams['xtick.major.width'] = 0
#        plt.rcParams['xtick.minor.width'] = 0
#        plt.rcParams['ytick.major.size'] = 0
#        plt.rcParams['ytick.minor.size'] = 0
#        plt.rcParams['ytick.major.width'] = 0
#        plt.rcParams['ytick.minor.width'] = 0
#        plt.rcParams['legend.frameon'] = False
#        plt.rcParams['legend.loc'] = 'center left'
#        plt.rcParams['axes.linewidth'] = 0

#        plt.gca().spines['right'].set_color('none')
#        plt.gca().spines['top'].set_color('none')
#        plt.gca().xaxis.set_ticks_position('bottom')
#        plt.gca().yaxis.set_ticks_position('left')
    else:
        plt.rcParams['figure.figsize'] = (4, 3)
        plt.rcParams['font.size'] = 10
#    plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
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
 

    ## excample
#init_plotting()
#
## begin subplots region
#plt.subplot(111)
#plt.gca().margins(0.1, 0.1)
#
#plt.plot(np.array(te_list),np.array(omega_star_fit_omega_star_1_chi2_list_te),linestyle ='-', marker='x', linewidth=1, color='r',label='Ti=20 eV')
#plt.plot(np.array(te_list),np.array(omega_star_fit_omega_star_1_chi2_list_teti),linestyle='--', marker='o',linewidth=1, color='b', label='Ti=Te')
#
##plt.gca().annotate(u'point $\\frac{\\tau}{2}$', xy=(x[2], y1[2]),  xycoords='data',
##                xytext=(30, -10), textcoords='offset points', size=8,
##                arrowprops=dict(arrowstyle='simple', fc='g', ec='none'))
#
#plt.xlabel(u'Te (eV)')
#plt.ylabel(r'$\omega_{\mathrm{fit}} / (\omega_*/(1+k_y^2\rho_s^2)) $',fontsize=1.5*plt.rcParams['font.size'])
##plt.title(u'Drift wave frequency')
#
#plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
#
##plt.subplot(122)
##plt.gca().margins(0.1, 0.1)
##plt.plot(x, y3, linestyle='--', marker='.', linewidth=1, color='g', label='sum')
##
##plt.gca().annotate(u'$y_x$', xy=(x[2], y3[2]),  xycoords='data',
##                xytext=(-30, -20), textcoords='offset points', size=8,
##                arrowprops=dict(arrowstyle='simple', fc='orange', ec='none'))
##
##plt.xlabel(u'x label')
##plt.ylabel(u'y label')
##plt.title(u'Second plot title')
##
##plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
## end subplots region
#
## output resulting plot to file
#plt.ylim(0.8,1.05)
#
#plt.tight_layout()
#plt.savefig('graph2.png')
#plt.savefig('graph2.eps')



