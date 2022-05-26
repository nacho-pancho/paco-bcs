#!/usr/bin/python3 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import util 
import paco.linop as linop
w = 8
W = np.eye(8*8)
D = np.copy(W)
linop.idct2d(W,8,8,D)
mosaic = util.dictionary_mosaic(D,2,0.25)
plt.imshow(mosaic,cmap=cm.gray)
plt.show()
