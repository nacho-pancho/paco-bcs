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
import argparse
import patch_mapping as patmap
import operators

rng = default_rng(1863699)
import util

#==============================================================================

DEF_WIDTH = 32
DEF_STRIDE = 16
DEF_USE_MMAP = False
DEF_WEPS = 1e-4
DEF_MAX_ITER = 500
DEF_MIN_CHANGE = 1e-5
DEF_ADMM_PENALTY = 1
DEF_REWEIGHT_ITER = 1

#==============================================================================
# GLOBAL VARIABLES
stride = None
width  = None
nrows  = None
ncols  = None

# ==============================================================================
ImAtAAtiA = None
AtAAti = None

def prox_f_block(D, P, b, w, args, y0=None):
    '''
    Computes the proximal operator for a single block
    The problem to be solved is:

    y* = arg min ||y||_p + 1/(2tau)|| y - w ||^2_2 s.t. Ay = b
    A = PDi

    :param Di: sparsifying operator matrix
    :param P: sampling  matrix
    :param B: compressed samples
    :param args: program arguments (includes tau)
    :param Y0: initial guess, if any
    :return y: the solution for this block
    '''
    #
    # we use a projected subgradient method
    #
    # Iterations:
    # 1) choose d(k) \in \partial|| y(k) ||_1 + ( y(k)- w )
    # 2) y(k+1) = Proj[ y(k) - s(k) d(k) ], where Proj(z) = z - At(AAt)i( Az - b )
    #                                                     = (I-At(AAt)iA) z + At(AAt)i b
    # this is not a descent method, so a monotonicity check needs to be done
    #
    # precomputed stuff:
    A = np.dot(P.T,Di.T)
    global ImAtAAtiA, AtAAti
    if ImAtAAtiA is None:
        AAt = np.dot(A,A.T)
        AAti= np.linalg.inv(AAt)
        AtAAti = np.dot(A.T,AAti)
        ImAtAAtiA = np.eye(len(w))-np.dot(AtAAti,A)

    m, k = P.shape
    if True: #y0 is None:
        # minimum norm solution of Ay = b: 
        y = np.dot(AtAAti,b)
    else:
        y = np.copy(y0)
    taup = args.tau # tau of the outer ADMM
    iter = 1
    s = 1 
    beta = 0.5
    while iter < args.inner_maxiter:
        # subgradient computation
        yprev = np.copy(y)
        d = np.sign(y) + (1/taup)*(y-w)
        # monotonicity backtracking a la Armijo
        f0 = np.sum(np.abs(y)) + (0.5/taup)*np.sum((y-w)**2)
        ycand = np.dot(ImAtAAtiA,y-s*d) + np.dot(AtAAti,b)
        fcand = np.sum(np.abs(ycand)) + (0.5/taup)*np.sum((ycand-w)**2)
        while fcand > f0 and s > 1e-5:
            s *= beta
            ycand = np.dot(ImAtAAtiA,y-s*d) + np.dot(AtAAti,b)
            fcand = np.sum(np.abs(ycand)) + (0.5/taup)*np.sum((ycand-w)**2)
        if fcand > f0:
            break
        y = ycand
        iter +=1
    #if ndy > args.inner_tol:
    #    print('x')
    #else:
    #    print('.')
    return y

# ==============================================================================

def prox_f(Di,P,B,W,args,Y0=None):
    '''
    Dispatches the proximal operator for each block (patch)

    :param Di: sparsifying operator matrix
    :param P: sampling  matrix
    :param B: compressed samples
    :param args: program arguments
    :param Y0: initial guess, if any
    :
    '''
    m = D.shape[1]
    n = B.shape[0]   
    Y = np.empty((n,m))
    for j in range(n):
        bj  = B[j,:]
        wj  = W[j,:]
        if Y0 is None:
            y0j = None
        else:
            y0j = Y0[j,:]
        Y[j,:] = prox_f_block(D, P, bj, wj, args, y0j)
    return Y

#==============================================================================

def prox_g(Z,tau):
    '''
    Projection onto patch consensus set
    '''
    I = patmap.stitch(Z, width, stride, nrows, ncols)
    return patmap.extract(I, width, stride, Z)

#==============================================================================

if __name__ == '__main__':
    epilog = "Output image file name is built from input name and parameters."
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--stride", type=int, default=12,
                        help="patch stride")
    parser.add_argument("-w", "--width", type=int, default=16,
                        help="patch stride")
    parser.add_argument("-t", "--tau", type=float, default=0.1,
                        help="ADMM penalty")
    parser.add_argument("-k", "--kappa", type=float, default=0.95,
                        help="Multiplier for diminishing ADMM stepsize.")
    parser.add_argument("-mi", "--maxiter", type=int, default=100,
                        help="Maximum ADMM iterations")
    parser.add_argument("-e", "--eps", type=float, default=1e-3,
                        help="Smoothing constant in L1 weights estimation.")
    parser.add_argument("-mc", "--minchange", type=float, default=1e-3,
                        help="Minimum change between ADMM iterations")
    parser.add_argument("-rw", "--rwiter", type=int, default=DEF_REWEIGHT_ITER,
                        help="How many L1 reweighting iterations to apply.")
    parser.add_argument("--proj-num", type=int, default=32,
                        help="Number of compressed sensing projections.")
    parser.add_argument("--proj-type", type=str, default="random",
                        help="Type of compressed sensing operator (random, binary, dct, idct).")
    parser.add_argument("--diff-type", type=str, default="uni",
                        help="Type of differential  operator (uni,bi).")
    parser.add_argument("--inner-tol", type=float, default=1e-3,
                        help="Tolerance for inner ADMM.")
    parser.add_argument("--outer-tol", type=float, default=1e-3,
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
    stride = args.stride

    Itrue = plt.imread(args.input)
    Itrue = patmap.pad_image(Itrue,width,stride)
    
    nrows,ncols = Itrue.shape
    npixels = nrows*ncols

    Xtrue = patmap.extract(Itrue,width,stride)

    n,m = Xtrue.shape
    print("Signal dimension =",m)
    print("Number of blocks =",n)
    P = operators.create_proj_op(m,args.proj_num,args.proj_type)
    K = P.shape[0]
    B = np.dot(Xtrue,P)
    nsamples = n*K
    cratio   = n*K / npixels
    print("Compression ratio (compressed samples / recovered samples)=",cratio)

    print("Computing difference operators")
    D = operators.create_sparse_op(width,width)
    Di = np.linalg.inv(D)
    print("Running algorithm")
    dif = 1e20
    Irec = np.empty(Itrue.shape)

    mosaic = util.dictionary_mosaic(D, 2, 0.25)
    plt.imshow(mosaic, cmap=cm.gray)
    plt.show()
    #
    # trick to treat w as vectors of size 2
    # each column corresponds to a pair h_i,v_i
    # and we can apply the vector soft thresholding to each column
    # and treat it as a vector when multiplying by G by just raveling
    #
    # B   = X*P
    # B*Pt = X*PPt
    # BPt(PPt)i = X
    print(P.shape)
    PtPPti = np.dot(P.T,np.linalg.inv(np.dot(P, P.T)+0.1*np.eye(P.shape[1]))) # a little Ridge reg.
    Xhat = np.dot(B,PtPPti) # first approximation using pseudoinverse

    Irec = patmap.stitch(Xhat, width, stride, nrows, ncols, Irec)
    Irec = np.minimum(1.0, np.maximum(0.0, Irec))
    iter = 0
    plt.imsave(f"iter{iter:04d}.png", Irec, cmap=cm.gray)

    Y = np.zeros((n,m))
    prevY = np.zeros(Y.shape)   # and its previous value
    Z = np.copy(Xhat)   # SECOND ADMM variable
    prevZ = np.copy(Z)   # previous value
    U = np.zeros(Z.shape)       # ADMM multipliers
    tau = args.tau              # ADMM penalty
    kappa = float(args.kappa)   # penalty increase factor
    Irec = np.empty(Itrue.shape) # reconstructed image
    #
    # main ADMM loop
    #
    for iter in range(1,args.maxiter):
        print(f"iter={iter:05}",end=", ")
        #
        # Y(k+1) <- prox_{tf}( Z(k) - U(k) )
        #
        np.copyto(prevY, Y)
        Y[:] = np.dot(prox_f(Di, P, B, np.dot(Z - U,D),args,Y0=prevY), Di)
        #
        # Z(k+1) <- prox_{tg}( Y(k+1) + U(k) )
        #
        np.copyto(prevZ, Z)
        Z[:] = prox_g(Y + U, tau)
        #
        # U(k+1) <- U(k) + Y(k+1) - Z(k+1)
        #
        dU = Y - Z
        U += dU
        #
        # convergence
        #
        ndU = np.linalg.norm(dU,'fro')/np.linalg.norm(U,'fro')       
        dX = np.linalg.norm(Y - prevY, 'fro') / (1e-10 + np.linalg.norm(Y, 'fro'))
        dW = np.linalg.norm(Z - prevZ, 'fro') / (1e-10 + np.linalg.norm(Z, 'fro'))
        F  = np.sum(np.abs(np.dot(Z,D)))
        Xerr = np.linalg.norm(Y - Xtrue, 'fro') / np.linalg.norm(Xtrue, 'fro')
        merr = np.linalg.norm(np.dot(Y, P) - B, 'fro') / np.linalg.norm(B)
        Irec = patmap.stitch(Y, width, stride, nrows, ncols, Irec)
        Ierr = np.sqrt(np.mean((Irec-Itrue)**2))
        psnr = 20*np.log10(1.0/Ierr)
        print(f"tau={tau:8.5f} dX={dX:8.5f} dW={dW:8.5f} dU={ndU:8.5f} Xerr={Xerr:8.5f} merr={merr:8.5f} cost={F:8.5f} IMAGE: MSE={Ierr:8.5f} PSNR={psnr:8.5f}")
        Irec = np.minimum(1.0,np.maximum(0.0,Irec))
        plt.imsave(f"iter{iter:04d}.png",Irec,cmap=cm.gray)
        if dX < args.outer_tol and dW < args.outer_tol:
            print('converged to tolerance.')
            break
        #if kappa < 1:
        #    tau = tau*kappa
    #
    # GUARDAR SALIDA
    #
    plt.imsave(args.output,Irec)
#==============================================================================
