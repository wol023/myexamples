import math
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

class SlopeFinder(object):
    def __init__(self, xdata=[], ydata=[], annotes=[], ax=None, xtol=None, ytol=None):
        if len(annotes) == 0:
            for idx in range(len(xdata)):
                annotes.append('(%g, %g)'%(xdata[idx],ydata[idx]))

        if (len(xdata)!=0) and (len(ydata)!=0):
            self.data = list(zip(xdata, ydata, annotes))
            if xtol is None:
                xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
            if ytol is None:
                ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
            self.xtol = xtol
            self.ytol = ytol
        else:
            self.xtol = xtol
            self.ytol = ytol

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.ax.autoscale(enable=False)
        self.drawnAnnotations = {}
        self.links = []
	self.visiblePoint_a = []
	self.visiblePoint_b = []
	self.visibleAnnotation_a = []
	self.visibleAnnotation_b = []
        self.xa = 0
        self.ya = 0
        self.annote_a = ''
        self.xb = 0
        self.yb = 0
        self.annote_b = ''
	self.shift_is_held = False
	self.slope_drawn = False

    def on_key_press(self, event):
        if event.key == 'shift':
           self.shift_is_held = True
    
    def on_key_release(self, event):
        if event.key == 'shift':
           self.shift_is_held = False

    def __call__(self, event):
   	tb = plt.get_current_fig_manager().toolbar
	if not self.shift_is_held:
   	    if event.button==1 and event.inaxes and tb.mode == '': # 3 is right click, 1 is left click
                self.xa = event.xdata
                self.ya = event.ydata
                self.annote_a='(%g, %g)'%(self.xa,self.ya)
                self.drawAnnote_a(event.inaxes, self.xa, self.ya, self.annote_a)
                
   	    if event.button==3 and event.inaxes and tb.mode == '': # 3 is right click, 1 is left click
                self.xb = event.xdata
                self.yb = event.ydata
                self.annote_b='(%g, %g)'%(self.xb,self.yb)
                self.drawAnnote_b(event.inaxes, self.xb, self.yb, self.annote_b)
        else: #delete annotation
   	    if event.button==1 and event.inaxes and tb.mode == '': # 3 is right click, 1 is left click
                self.xa = []
                self.ya = []
                self.annote_a=[]
                if len(self.visibleAnnotation_a)!=0:
                    markers = self.visibleAnnotation_a
                    for m in markers:
                        m.remove()
                    self.visibleAnnotation_a=[]
                    self.visiblePoint_a=[]
                    self.ax.figure.canvas.draw_idle()
                self.drawSlope(self.ax)
   	    if event.button==3 and event.inaxes and tb.mode == '': # 3 is right click, 1 is left click
                self.xb = []
                self.yb = []
                self.annote_b=[]
                if len(self.visibleAnnotation_b)!=0:
                    markers = self.visibleAnnotation_b
                    for m in markers:
                        m.remove()
                    self.visibleAnnotation_b=[]
                    self.visiblePoint_b=[]
                    self.ax.figure.canvas.draw_idle()
                self.drawSlope(self.ax)



    def drawAnnote_a(self, ax, x, y, annote):
        #print 'drawAnnote a'
        if len(self.visiblePoint_a)!=0:
            markers = self.visibleAnnotation_a
            for m in markers:
                m.set_visible(not m.get_visible())
            self.visibleAnnotation_a=[]
            self.visiblePoint_a=[]
        t = ax.text(x, y, " - %s" % (annote),)
        m = ax.scatter([x], [y], marker='d', c='r', zorder=100)
        self.ax.figure.canvas.draw_idle()
        self.visibleAnnotation_a = (t, m)
        if m.get_visible():
            self.visiblePoint_a.append([x,y])
        self.drawSlope(ax)
            
    def drawAnnote_b(self, ax, x, y, annote):
        #print 'drawAnnote b'
        if len(self.visiblePoint_b)!=0:
            markers = self.visibleAnnotation_b
            for m in markers:
                m.set_visible(not m.get_visible())
            self.visibleAnnotation_b=[]
            self.visiblePoint_b=[]
        t = ax.text(x, y, " - %s" % (annote),)
        m = ax.scatter([x], [y], marker='d', c='g', zorder=100)
        self.ax.figure.canvas.draw_idle()
        self.visibleAnnotation_b = (t, m)
        if m.get_visible():
            self.visiblePoint_b.append([x,y])
        self.drawSlope(ax)


    def drawSlope(self, ax):
        if len(self.visiblePoint_a)+len(self.visiblePoint_b)>=2:
                if self.slope_drawn:
                    if len(ax.lines)!=0:
                        del ax.lines[-1]
                        ax.linetexts.remove()
                        self.slope_drawn=False
                if ax.get_yscale()=='log' and ax.get_xscale()=='log':
                    xx = np.linspace(self.xa,self.xb)
                    slope = (np.log10(self.yb)-np.log10(self.ya))/(np.log10(self.xb)-np.log10(self.xa))
                    y_cross =np.log10(self.ya)-slope*np.log10(self.xa)
                    yy = 10**(slope*(np.log10(xx)-np.log10(self.xa))+np.log10(self.ya))
                    ax.linetexts=ax.annotate('a=%g,b=%g'%(slope,y_cross),xy=( 10**(0.5*(np.log10(self.xa)+np.log10(self.xb))), 10**(slope*(np.log10(self.xa)+np.log10(self.xb))/2+y_cross)), xytext=( (10**(0.5*(np.log10(self.xa)+np.log10(self.xb))), 10**(slope*(np.log10(self.xa)+np.log10(self.xb))/2+y_cross)))  )
                elif ax.get_yscale()=='log' and ax.get_xscale()=='linear':
                    xx = np.linspace(self.xa,self.xb)
                    slope = (np.log10(self.yb)-np.log10(self.ya))/(self.xb-self.xa)
                    y_cross =np.log10(self.ya)-slope*self.xa
                    yy = 10**(slope*(xx-self.xa)+np.log10(self.ya))
                    ax.linetexts=ax.annotate('a=%g,b=%g'%(slope,y_cross),xy=(0.5*(self.xa+self.xb), 10**(slope*(self.xa+self.xb)/2+y_cross)), xytext=(0.5*(self.xa+self.xb), 10**(slope*(self.xa+self.xb)/2+y_cross))  )
                elif ax.get_yscale()=='linear' and ax.get_xscale()=='log':
                    xx = np.linspace(self.xa,self.xb)
                    slope = ((self.yb)-(self.ya))/(np.log10(self.xb)-np.log10(self.xa))
                    y_cross =(self.ya)-slope*(np.log10(self.xa))
                    yy = (slope*(np.log10(xx)-np.log10(self.xa))+(self.ya))
                    ax.linetexts=ax.annotate('a=%g,b=%g'%(slope,y_cross),xy=( 10**(0.5*(np.log10(self.xa)+np.log10(self.xb))), (slope*((np.log10(self.xa)+np.log10(self.xb))/2)+y_cross)), xytext=(10**(0.5*(np.log10(self.xa)+np.log10(self.xb))), (slope*((np.log10(self.xa)+np.log10(self.xb))/2)+y_cross))  )
                else:
                    xx = np.linspace(self.xa,self.xb)
                    slope = (self.yb-self.ya)/(self.xb-self.xa)
                    y_cross =self.ya-slope*self.xa
                    yy = slope*(xx-self.xa)+self.ya
                    ax.linetexts=ax.annotate('a=%g,b=%g'%(slope,y_cross),xy=(0.5*(self.xa+self.xb), 0.5*(self.ya+self.yb)), xytext=(0.5*(self.xa+self.xb), 0.5*(self.ya+self.yb))  )

                ax.plot(xx,yy,'r--') 
                self.slope_drawn=True
        else:
            if self.slope_drawn:
                if len(ax.lines)!=0:
                    del ax.lines[-1]
                    ax.linetexts.remove()
                    self.slope_drawn=False
                    self.ax.figure.canvas.draw_idle()


#x = np.linspace(0,1,10)
#y = x*x
#annotes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
#
#fig, ax = plt.subplots()
#ax.plot(x,y)
#ax.set_xscale('log')
#ax.set_yscale('log')
##ax.scatter(x,y)
###af =  SlopeFinder(x,y, annotes, ax=ax)
#af =  SlopeFinder(x,y, ax=ax)
#fig.canvas.mpl_connect('button_press_event', af)
#fig.canvas.mpl_connect('key_press_event', af.on_key_press)
#fig.canvas.mpl_connect('key_release_event', af.on_key_release)
#plt.show()


#def linkAnnotationFinders(afs):
#  for i in range(len(afs)):
#    allButSelfAfs = afs[:i]+afs[i+1:]
#    afs[i].links.extend(allButSelfAfs)
#
#plt.subplot(121)
#plt.scatter(x,y)
#af1 = DataFinder(x,y, annotes)
#plt.connect('button_press_event', af1)
#
#plt.subplot(122)
#plt.scatter(x,y)
#af2 = DataFinder(x,y, annotes)
#plt.connect('button_press_event', af2)
#linkAnnotationFinders([af1, af2])
#
#plt.show()
