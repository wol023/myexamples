import numpy as np
import matplotlib.pyplot as plt

from setupplot import init_plotting


from scipy.special import erf, erfc


x = np.linspace(0, 4.5, 64)


init_plotting()
ax=plt.subplot(111)
plt.gca().margins(0.1, 0.1)

plt.gca().set_yscale('log')

plt.plot(x,(erfc(x)*100) ,linestyle='-',linewidth=1,color='b',label='erfc'+r'($V_{max}/\tilde{v})$')
#plt.scatter(xcm[x_point_index_in_plot],yy[x_point_index_in_plot],marker="o",linewidth=1,color='g',label='measured point' )
plt.xlabel(r'$V_{max}/\tilde{v}$')
plt.ylabel(u'Error (%)')
#plt.ylabel(r'$\omega_{\mathrm{fit}} / (\omega_*/(1+k_y^2\rho_s^2)) $',fontsize=1.5*plt.rcParams['font.size'])
#plt.title(u'Drift wave frequency')

plt.grid()
#plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
plt.gca().legend(loc='best',frameon=1)

## output resulting plot to file
plt.xlim(min(x),max(x))
#

plt.tight_layout()
#plt.show()

plt.savefig('foo1.png')
plt.savefig('foo1.tiff')
plt.savefig('foo1.eps')
#plt.close('all')
#plt.clf()



#import pylab
#import Image
#init_plotting('2x3')
#f = pylab.figure()
#for n, fname in enumerate(('foo1.png', 'foo2.png', 'foo3.png', 'foo4.png', 'foo5.png','foo6.png')):
#     image=Image.open(fname)#.convert("L")
#     arr=np.asarray(image)
#     ax=f.add_subplot(2, 3, n+1)
#     ax.axis('off')
#     pylab.imshow(arr)
#pylab.tight_layout()
#pylab.savefig('foo0.png')
#pylab.show()



