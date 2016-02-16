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

#read history file
x_list=[]
y_list=[]
with open("potential_hist_1.curve", 'r') as f:
    for line in f:
        if line.lstrip().startswith('#'): #skip comment
            continue
        line =line.rstrip() #skip blank line
        if not line:
            continue 
        else: #noncomment line
            strippedline=line
            lhsrhs = strippedline.split(" ")
            l=0
            while l<len(lhsrhs): #strip white spaces in lhs
                lhsrhs[l]=lhsrhs[l].rstrip()
                lhsrhs[l]=lhsrhs[l].lstrip()
                l=l+1
            x_list.append(float(lhsrhs[0]))
            y_list.append(float(lhsrhs[1]))

f.closed

del x_list[-10:]
del y_list[-10:]

#make time unit to second/2/pi
#print type(x_list)
x_list[:] = [i*ref_time/2.0/np.pi for i in x_list] 
#print x_list

# number of signal points
#N = 400
#T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N)
#y = np.exp(50.0 * 1.j * 2.0*np.pi*x) #+ 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)

N = len(x_list)
T = x_list[len(x_list)-1] / N
x = np.array(x_list)
y = np.array(y_list)

yf = scipy.fftpack.fft(y)
xf = scipy.fftpack.fftfreq(N, T)

xf = scipy.fftpack.fftshift(xf)
yplot = scipy.fftpack.fftshift(yf)

print np.abs(yplot).argmax()
freqmax=xf[np.abs(yplot).argmax()]
print 'maximum frequency =', freqmax,'[Hz]'
yv = np.real(y.max()*np.exp(-freqmax*1.j*2.0*np.pi*x))


yfv = scipy.fftpack.fft(yv)
xfv = scipy.fftpack.fftfreq(N, T)

xfv = scipy.fftpack.fftshift(xfv)
yplotv = scipy.fftpack.fftshift(yfv)




fig, ax = plt.subplots(2, 1)
ax[0].plot(x,y,'xb-')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')

ax[0].plot(x,yv,'.r-')

ax[1].plot(xfv,1.0/N * np.abs(yplot),'.b-') # plotting the frequency spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

ax[1].plot(xfv,1.0/N * np.abs(yplotv),'.r-')

plt.show()
