#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PACO Block Compressed Sensing
@author: nacho
"""
import os
import time
import sys
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import cs_itv as cs

DEF_TIME_STRIDE = 1
DEF_SPACE_STRIDE = 2
DEF_BLOCK_SIZE = 32
DEF_USE_MMAP = False
DEF_WEPS = 1e-4
DEF_MAX_ITER = 500
DEF_MIN_CHANGE = 1e-5
DEF_ADMM_PENALTY = 5
DEF_REWEIGHT_ITER = 1
DEF_NUM_PROJ = 0

def compute_differential_operator(nrows,ncols):
    N = nrows*ncols
    Dh = np.eye(N,N)
    li = 0
    for i in range(nrows):
        for j in range(ncols):
            if j > 0:
                Dh[li,i*ncols+(j-1)] = -1
            li = li+1
    Dv = np.eye(N,N)
    li = 0
    for i in range(nrows):
        for j in range(ncols):
            if i > 0:
                Dv[li,(i-1)*ncols+j] = -1
            li = li+1
    D = np.concatenate((Dh,Dv))
    return D

DtD  = None
DtDi = None
PDtDi = None
PDtDiDt = None
PDtDiPt = None
PDtDiPti = None

def solve_prox_f(D,w,P,y,tau):
    global DtD
    global DtDi
    global PDtDi
    global PDtDiDt
    global PDtDiPt
    global PDtDiPti
    # 1) x(k+1) = argmin_{x} Dt(Dx - w(k)) = 0 s.t.  Wx = y =>   
    # solution
    # min (1/2)||Dx-w||_2^2 s.t. Px=y
    Dtw  = np.dot(D.T,w)
    if DtD is None:
        DtD   = np.dot(D.T,D)
        DtDi = np.linalg.inv(DtD)
        PDtDi   = np.dot(P,DtDi)
        PDtDiDt = np.dot(PDtDi,D.T)
        PDtDiPt = np.dot(PDtDi,P.T)
        PDtDiPti = np.linalg.inv(PDtDiPt)
    
    PDtDiDtw = np.dot(PDtDi,Dtw)
    L = np.dot( PDtDiPti, PDtDiDtw - y )
    Ptl = np.dot(P.T,L)
    x = np.dot( DtDi, Dtw - Ptl )
    return x

def solve_prox_g(z,tau):
    N2 = len(z)
    N = int(N2/2)
    z = np.reshape(z,(2,N))
    nz = np.sqrt(np.sum(z**2,axis=0))
    tz = np.outer(np.ones((2,1)),np.maximum(0,1-tau/(nz+1e-10)))
    w = np.reshape(tz*z,(2*N))
    return w


def cs_itv(D,P,y,tau,x0):
    m = D.shape[1]
    x = np.zeros(m)
    z = rng.normal(size=(2*m))
    u = np.zeros(2*m)
    prevx = np.zeros(m)
    for iter in range(maxiter):
        #print("iter",iter,end=", ")
        np.copyto(prevx,x)

        x = solve_prox_f( D, z-u, P, y, tau )
        z = solve_prox_g( D.dot(x) + u, tau )
        u = u + D.dot(x) - z        

        dif = np.linalg.norm(x-prevx)/(eps+np.linalg.norm(x))
        err = np.linalg.norm(x-0)/np.linalg.norm(x0)
        merr = np.linalg.norm(np.dot(P,x)-y)/np.linalg.norm(y)
        #if np.mod(iter,10) == 0:
        #print("tau=",tau,"dif=",dif,"err=",err,"merr=",merr)
        #plt.imsave(f"iter{iter:04d}.png",np.reshape(x,(nrows,ncols)),cmap=cm.gray)
        #if kappa < 1:
        #    tau = tau*kappa
    return x


if __name__ == '__main__':
    epilog = "Output image file name is built from input name and parameters."
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--block-size", type=int, default=DEF_BLOCK_SIZE,
                        help="Block size")
    parser.add_argument("-t", "--tau", type=float, default=DEF_ADMM_PENALTY,
                        help="ADMM penalty")
    parser.add_argument("-k", "--kappa", type=float, default=0.95,
                        help="Multiplier for diminishing ADMM stepsize.")
    parser.add_argument("-m", "--maxiter", type=int, default=DEF_MAX_ITER,
                        help="Maximum ADMM iterations")
    parser.add_argument("-e", "--eps", type=float, default=DEF_WEPS,
                        help="Smoothing constant in L1 weights estimation.")
    parser.add_argument("-mc", "--minchange", type=float, default=DEF_MIN_CHANGE,
                        help="Minimum change between ADMM iterations")
    parser.add_argument("-rw", "--rwiter", type=int, default=DEF_REWEIGHT_ITER,
                        help="How many L1 reweighting iterations to apply.")
    parser.add_argument("-p", "--nproj", type=int, default=0,
                        help="Number of compressed sensing projections.")
    
    parser.add_argument("input", help="input image file")
    #parser.add_argument("smat", help="Sensing matrix")
    parser.add_argument("output", default="", help="recovered image file")
    args = parser.parse_args()

    cmd = " ".join(sys.argv)
    print(("Command: " + cmd))
    #
    #
    # parametros (por ahora mejores para desfile1 por lo menos)
    #
    params = {'maxiter': args.maxiter,
              'tau':args.tau,
              'kappa':args.kappa,
              'weps':args.eps,
              'minchange':args.minchange,
              'rwiter':args.rwiter,
              'output':args.output}

    I0 = plt.imread(args.input)
    if len(I0.shape) > 2:
        I0 = np.mean(I0,axis=2)
    nrows,ncols = I0.shape
    w =  args.block_size
    mg = int(nrows/w)
    ng = int(nrows/w)
    nrows = mg*w
    ncols = ng*w
    I0 = I0[:nrows,:ncols]
    m = w*w
    n = ng*mg # number of blocks
    Irec = np.empty((nrows,ncols))
    #
    # RANDOM ORTHOGONAL SENSING MATRIX
    #
    print("Block dimension m=",m)
    print("Number of blocks n=",n)
    rng = default_rng(1863699)
    print("Sampling random matrix")
    Wn = rng.normal(size=(m,m))
    if args.nproj > 0:
        p = args.nproj
    else:
        p = int(m/5)
    print("Choosing ",p,"rows")
    Pk = Wn[:p,:]
    print("Orthogonalization")
    _,_,P = np.linalg.svd(Pk,full_matrices=False)
    
    print("Compute measurements")
    X0 = np.empty((n,m))
    Y  = np.empty((n,p))
    li = 0
    for i in range(0,nrows,w):
        for j in range(0,ncols,w):
            X0[li,:] = I0[i:(i+w),j:(j+w)].ravel()
            Y[li,:] = np.dot(P,X0[li,:])
            li += 1
    print("Computing difference operators")
    D = cs.compute_differential_operator(w,w)
    
    #
    # CORRER  ALGORITMO
    #
    # Need to solve min_{h,v,x} \sum_{i=1}^{N} ||(h_i,v_i)||_2 + 
    #                           (1/2L)||Dhx-h||_2^2 + 
    #                           (1/2L)||Dvx-v||_2^2 + 
    #                           (1/2L)||Wx-y||_2^2  s.t. Dhx=h Dvx=v Wx=y
    #
    #             = min_{h,v,x} \sum_{i=1}^{N} ||(h_i,v_i)||_2 + (1/2L)||[Dh] x - [h]||_2^2  s.t. [Dh]x = [h]  
    #                                                                  ||[Dv]   - [v]||_2^2       [Dv] = [v]  
    #                                                                                              Wx   = y
    #             = min_{w,x} \sum_{i=1}^{N} ||(h_i,v_i)||_2 + (1/2L)||D x -w||_2^2  s.t. D x = w  
    #                                                                                     Wx = y
    #
    # ADMM
    # 1) x(k+1) = argmin_{x} Dt(Dx - w(k)) = 0 s.t.  Wx = y =>   
    # solution
    # x = (DtD)iDt z for z = (w(k)-u(k))
    # 2) w(k+1) = argmin_{w} \sum_{i=1}^{N} ||(w_i,w_{N+i})||_2 + (1/2L)||Dz - w||_2^2 for z = x(k+1)-u(k)
    # 3) u(k+1) = u(k) + Dx(k+1) - w(k+1)

    print("Running algorithm")
    dif = 1e20
    # trick to treat w as vectors of size 2
    # each column corresponds to a pair h_i,v_i
    # and we can apply the vector soft thresholding to each column
    # and treat it as a vector whenmultiplying by G by just raveling
    tau = args.tau
    eps = 1e-5
    X = np.zeros(X0.shape)
    maxiter = args.maxiter
    kappa = float(args.kappa)
    li = 0
    for i in range(0,nrows,w):
        for j in range(0,ncols,w):
            y = Y[li,:]
            x0 = X0[li,:]
            print(f"block ({i},{j}) ({li} of {n})")
            X[li,:] = cs.cs_itv(D,P,y,x0)
            # stitch
            print(Irec.shape,X0.shape,i,j,li)
            Irec[i:(i+w),j:(j+w)] = np.reshape(X[li,:],(w,w)) 
            Irec = np.minimum(1.0,np.maximum(0.0,Irec))
            li += 1
            plt.imsave(f"Irec.png",Irec,cmap=cm.gray)
    #
    # GUARDAR SALIDA
    #
    #plt.imsave(args.output,Irec)
