#!/usr/bin/env python3
import numpy as np
import glob
import matplotlib.pyplot as plt

files = glob.glob("*_py.txt")
for f in files:
    print(f)
    pymat = np.loadtxt(f)
    cmat  = np.loadtxt(f[:-6]+"c.txt")
    maxa = np.max(np.abs(pymat-cmat))
    print('maxa',maxa,'mean',np.mean(pymat-cmat),'rmse',np.sqrt(np.mean((pymat-cmat)**2)))
    if maxa > 1e-5:
        plt.figure(figsize=(10,10))
        plt.imshow(pymat-cmat)
        plt.show()
