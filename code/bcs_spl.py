#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#==============================================================================

"""
Block Compressed Sensing
Smoothed Projected Landweber
@author: nacho
"""
import sys
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import patch_mapping as patmap
import paco.linop as linop

import scipy.signal as dsp

rng = default_rng(1863699)

#==============================================================================

def create_proj_op(m,nproj,type='random'):
    if nproj > 0:
        K = nproj
    else:
        K = int(m/5)
    print("Taking ",K," measurements per block")
    if type == 'random' or type == 'binary':
        Wn = rng.normal(size=(m,m))
        # EXPERIMENTAL: add DC term
        Wn[0,:] = 1.0/np.sqrt(m)
        Pk = Wn[:K,:]
        _,_,P = np.linalg.svd(Pk,full_matrices=False)
        if type == 'binary':
            P = 1-2*(P>0)
        #
        # normalize
        #
        nP = np.sum(P**2,axis=1)
        W = np.diag(1.0/np.sqrt(nP))
        P = np.dot(W,P)
        return P
    else:
        return None


#==============================================================================

if __name__ == '__main__':
    epilog = "Output image file name is built from input name and parameters."
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--width", type=int, default=16,
                        help="patch width")
    parser.add_argument("--lam", type=float, default=6,
                        help="Thresholding parameter")
    parser.add_argument("--kappa", type=float, default=0.6,
                        help="Thresholding parameter decrease (0.6 from paper)")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Maximum ADMM iterations")
    parser.add_argument("-e", "--eps", type=float, default=1e-5,
                        help="Smoothing constant in L1 weights estimation.")
    parser.add_argument("--proj-num", type=int, default=50,
                        help="Number of compressed sensing projections.")
    parser.add_argument("--proj-type", type=str, default="random",
                        help="Type of compressed sensing operator (random, binary, dct, idct).")
    parser.add_argument("--inner-tol", type=float, default=5e-4,
                        help="Tolerance for inner ADMM.")
    parser.add_argument("--outer-tol", type=float, default=5e-4,
                        help="Tolerance for outer ADMM.")
    parser.add_argument("--inner-maxiter", type=int, default=100,
                        help="Maximum number of iterations in inner ADMM.")

    parser.add_argument("input", help="input image file")
    parser.add_argument("output", default="", help="recovered image file")
    args = parser.parse_args()

    cmd = " ".join(sys.argv)
    print(("Command: " + cmd))
    print(args)
    #
    #
    # parametros (por ahora mejores para desfile1 por lo menos)
    #
    # buenos parametros: 4x8x8+1+2+#
    width  = args.width
    stride = width # args.stride

    Itrue = plt.imread(args.input)
    Itrue = patmap.pad_image(Itrue,width,stride)
    M,N = Itrue.shape
    nrows,ncols = Itrue.shape
    npixels = nrows*ncols

    Xtrue = patmap.extract(Itrue,width,stride)

    n,m = Xtrue.shape
    print("Signal dimension =",m)
    print("Number of blocks =",n)
    P = create_proj_op(m,args.proj_num,args.proj_type)
    aux = np.eye(m)
    w   = int(np.sqrt(m))
    D   = np.empty((m,m))
    linop.dct2d(aux,w,w,D)
    G   = np.dot(D.T,D)
    W   = np.diag(1.0/np.diag(np.sqrt(G)))
    D   = np.dot(D,W)
    G   = np.dot(D.T,D)

    K = P.shape[0]
    B = np.dot(Xtrue,P.T)
    nsamples = n*K
    cratio   = n*K / npixels
    print("Compression ratio (compressed samples / recovered samples)=",cratio)

    print("Running algorithm")
    dif = 1e20
    I = np.zeros(Itrue.shape)
    Y = np.zeros((n,m))
    Z = np.zeros((n,m))
    prevY = np.zeros(Y.shape)   # and its previous value
    J = np.empty(Itrue.shape) # reconstructed image
    Rxx = 0.01*np.eye(m)
    Y = np.dot(np.dot(B,P),np.linalg.inv(np.dot(P.T,P)+Rxx))
    I = patmap.stitch(Y,width,stride,nrows,ncols)
    plt.imsave('0.png',I)
    #
    # main ADMM loop
    #
    lam = args.lam
    for i in range(args.max_iter):
        # smoothing (on whole image)
        I = dsp.wiener(I,(3,3))
        # Y = patches(I)
        Y[:] = patmap.extract(I, width, stride)
        # landweber iteration (on patches)
        Y = Y + np.dot(B - np.dot(Y,P.T),P)
        #
        # projection
        #
        Z = np.dot(Y,D.T)
        sigma = np.median(np.abs(Z))/0.6745
        tau = lam * sigma * np.sqrt(2*np.log(M*N))
        Z[np.abs(Z) < tau] = 0
        Y = np.dot(Z,D)
        #
        # another landweber iteration
        #
        Y = Y + np.dot(B - np.dot(Y,P.T),P)
        #
        # stitch
        #
        I = patmap.stitch(Y,width,stride,nrows,ncols)
        dY = np.linalg.norm(Y - prevY, 'fro') / (1e-10 + np.linalg.norm(Y, 'fro'))
        #
        # reduce lambda (paper does it)
        #
        lam *= args.kappa
        Xerr = np.linalg.norm(Y - Xtrue, 'fro') / np.linalg.norm(Xtrue, 'fro')
        merr = np.linalg.norm(np.dot(Y, P.T) - B, 'fro') / np.linalg.norm(B)
        Ierr = np.sqrt(np.mean((I-Itrue)**2))
        psnr = 20*np.log10(1.0/Ierr)
        print(f"i={i:5} dX={dY:8.5f} MSE={Ierr:8.5f} PSNR={psnr:8.5f}")
        I = np.minimum(1.0,np.maximum(0.0,I))
        plt.imsave(f"iter{i:04d}.png",I,cmap=cm.gray)
        if dY < args.outer_tol:
            print('converged to tolerance.')
            break
    #
    # GUARDAR SALIDA
    #
    plt.imsave(args.output,I,cmap=cm.gray)
#==============================================================================
