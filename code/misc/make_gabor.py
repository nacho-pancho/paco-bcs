#!/usr/bin/python3 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import util 
w = 10
D = None
#
# make a huge gabor dictionary
#
xs = np.arange(0,w,step=2)
ys = np.arange(0,w,step=2)
ps = range(3)
angs = np.arange(0,np.pi/2,np.pi/8)
natoms = len(xs)*len(ys)*len(ps)*len(angs)

def render_atom(w,x,y,s,a):
    y = np.zeros((w,w))
    return y

k = 0
D = np.empty((natoms,w*w))
for x in xs:
    for y in ys:
        for period in ps:
            s = 2*period
            for a in angs:
                render_atom(w,x,y,s,a)
                k += 1

mosaic = util.dictionary_mosaic(D,2,0.25)
plt.imshow(mosaic,cmap=cm.gray)
plt.show()
