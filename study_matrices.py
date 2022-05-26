#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#==============================================================================

"""
PACO Block Compressed Sensing
@author: nacho
"""
import sys
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import paco_bcs


#==============================================================================

if __name__ == '__main__':
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    mn = m*n
    p = int(sys.argv[3])
    D = paco_bcs.create_diff_op(m,n)
    P = paco_bcs.create_proj_op(mn,p,type='binary')
    print('D',D)
    print('P',P)
    G     = np.dot(D.T,D)
    print('G',G)
    Gi    = np.linalg.inv( G + np.eye(G.shape[0]) )
    print('Gi',Gi)
    PGi   = np.dot(P,Gi)
    print('PGi',PGi)
    PGiPt =  np.dot(PGi,P.T)
    print('PGiPt',PGiPt)
    PGiPti =  np.linalg.inv(PGiPt)
    print('PGiPti',PGiPti)
    Ca = Gi - np.dot(PGi.T,np.dot(PGiPti,PGi))
    print('Ca',Ca)
    Cb = np.dot(PGi.T,PGiPti)
    print('Cb',Cb)

#==============================================================================
