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
import paco.linop as linop
import paco.ssim as ssim
import paco.core as core
import operators

rng = default_rng(1863699)
use_uzawa = True

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
uzawa_mu = None

def prox_g(Yu,tau, args, Y):
    '''
    :
    :param Yu: Uzawa's argument to proximal operator: Y - mu/lambda*Dt(DY-Z+U)
    :param Y: output
    :param args: program args
    '''
    n,m = Yu.shape
    Yu = np.reshape(Yu,(-1,2)) # a lot of length-2 rows
    nY  = np.sqrt(np.sum(Yu**2,axis=1)) # sum them mofo row norms
    weights = np.ones(len(nY))
    for i in range(args.rwiter):
        wtau = tau*weights
        #tnY = np.array( [(1-tau/nj) if nj > tau else 0 for nj in nY] )
        tnY = 1-wtau/np.maximum(wtau,nY)
        eps = 1e-5+np.mean(tnY)
        weights = eps + tnY

    tY = Yu*np.outer( tnY, np.ones((2,1)) )
    Y[:] = np.reshape(tY,(n,m))


#==============================================================================

PtPPti = None
ImPtPPtiP = None
POCSimg = None
def prox_f_uzawa(YpU,P,B,args,Z):
    '''
    POCS for projecting onto the intersection of the consensus set and the CS set
    Z <- Z - Pt(PPt)i(PZ - B)
    written in transposed form (B = ZP, BPt = ZPPt, etc...)
    '''
    global PtPPti, ImPtPPtiP, POCSimg
    if ImPtPPtiP is None:
        PtPPti = np.dot(P.T,np.linalg.inv(np.dot(P,P.T)))
        ImPtPPtiP = np.dot(PtPPti,P)
        ImPtPPtiP = np.eye(m) - ImPtPPtiP
    # consensus
    Z2 = np.copy(YpU)
    Z1 = np.copy(Z)
    iter = 0
    while iter < args.inner_maxiter:
        Z2prev = np.copy(Z2)
        if POCSimg is None:
            POCSimg = patmap.stitch(Z2, width, stride, nrows, ncols)
        else:
            patmap.stitch(Z2, width, stride, nrows, ncols,POCSimg)
        #
        # Z1 is projected onto consensus set
        #
        patmap.extract(POCSimg, width, stride, Z1)
        #
        # Z2 is projected onto compressive sensing set
        # compressive sensing Z2/PZ2=B: Z2 = Z2 - PtPPt(PZ2-B)
        #
        Z2 = np.dot(Z1,ImPtPPtiP.T) + np.dot(B,PtPPti.T)
        #
        # convergence metrics
        #
        dZ2 = np.linalg.norm(Z2prev-Z2,'fro')/ np.prod(Z2.shape)
        #Z2viol = np.linalg.norm(Z2-Z1,'fro') / np.prod(Z2.shape)
        #Z1viol = np.linalg.norm(np.dot(Z1, P.T) - B, 'fro') / np.prod(B.shape)
        #print('\titer',iter,' violation CZ=',Z2viol,'PZ-B=',Z1viol)
        # variant: mid point POCS
        #Z2[:] = 0.5*(Z2+Z1)
        if dZ2 < args.inner_tol:
            break
        iter += 1
    np.copyto(Z,Z2)

#==============================================================================

if __name__ == '__main__':
    epilog = "Output image file name is built from input name and parameters."
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--stride", type=int, default=DEF_STRIDE,
                        help="patch stride")
    parser.add_argument("-w", "--width", type=int, default=DEF_WIDTH,
                        help="patch stride")
    parser.add_argument("-t", "--tau", type=float, default=DEF_ADMM_PENALTY,
                        help="ADMM penalty")
    parser.add_argument("--mu", type=float, default=0.5,
                        help="uzawas mu")
    parser.add_argument("-k", "--kappa", type=float, default=0.95,
                        help="Multiplier for diminishing ADMM stepsize.")
    parser.add_argument("-e", "--eps", type=float, default=DEF_WEPS,
                        help="Smoothing constant in L1 weights estimation.")
    parser.add_argument("--rwiter", type=int, default=DEF_REWEIGHT_ITER,
                        help="How many L1 reweighting iterations to apply.")
    parser.add_argument("--proj-num", type=int, default=0,
                        help="Number of compressed sensing projections.")
    parser.add_argument("--proj-type", type=str, default="random",
                        help="Type of compressed sensing operator (random, binary, dct, idct).")
    parser.add_argument("--diff-type", type=str, default="uni",
                        help="Type of differential  operator (uni,bi).")
    parser.add_argument("--inner-tol", type=float, default=1e-5,
                        help="Tolerance for inner ADMM.")
    parser.add_argument("--outer-tol", type=float, default=1e-4,
                        help="Tolerance for outer ADMM.")
    parser.add_argument("--inner-maxiter", type=int, default=100,
                        help="Maximum number of iterations in inner iter.")
    parser.add_argument("--maxiter", type=int, default=100,
                        help="Maximum number of iterations in ADMM.")

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
    mask = np.zeros((1,*Itrue.shape))
    Xtrue = patmap.extract(Itrue,width,stride)

    n,m = Xtrue.shape
    print("Signal dimension =",m)
    print("Number of blocks =",n)
    P = operators.create_proj_op(m,args.proj_num,args.proj_type)
    print(P.shape)
    K = P.shape[0]
    B = np.dot(Xtrue,P.T)
    nsamples = n*K
    cratio   = n*K / npixels
    print("Compression ratio (compressed samples / recovered samples)=",cratio)

    print("Computing difference operators")
    D =     operators.create_diff_op(width,width,args.diff_type)
    print("Running algorithm")
    dif = 1e20
    Irec = np.empty(Itrue.shape)

    #
    # trick to treat w as vectors of size 2
    # each column corresponds to a pair h_i,v_i
    # and we can apply the vector soft thresholding to each column
    # and treat it as a vector when multiplying by G by just raveling
    #
    # B   = PX
    # PtB = PtPX
    # (PtP)iPtB = X
    PtP = np.dot(P.T, P)
    PtPi = np.linalg.inv(PtP+0.1*np.eye(m))
    PtPiPt = np.dot(PtPi,P.T) # a little Ridge reg.

    Y = np.dot(B,PtPiPt.T)   # SECOND ADMM variable
    prevY = np.zeros(Y.shape)   # and its previous value
    Z = np.zeros((n,2*m)) # first approximation using pseudoinverse
    prevZ = np.zeros(Z.shape)   # previous value
    U = np.zeros(Z.shape)       # ADMM multipliers
    tau = args.tau              # ADMM penalty
    maxiter = args.maxiter      # maximum number of iterations
    kappa = float(args.kappa)   # penalty increase factor
    Irec = np.empty(Itrue.shape) # reconstructed image
    #
    # main Linearized ADMM loop
    #
    # min f(Y) + g(DY)
    # min f(Y) + g(Z) s.t. DY=Z
    #
    print(np.linalg.norm(D)**2)
    uzawa_mu = args.mu * args.tau / (np.linalg.norm(D)**2)
    use_uzawa = args.mu > 0
    #uzawa_mu = 0.9 * args.tau / (np.linalg.norm(D))
    print('tau',args.tau, 'mu',args.mu, 'uzawa mu',uzawa_mu)
    nm = n * m
    tau = args.tau
    for iter in range(maxiter):
        #
        # Y(k+1) <- prox_{tf}( Z(k) - U(k) )
        #
        np.copyto(prevY, Y)
        prox_f_uzawa(Y - (uzawa_mu / tau) * np.dot(np.dot(Y, D.T) - (Z - U), D) , P, B, args, Y)
        #
        # Z(k+1) <- prox_{tg}( Y(k+1) + U(k) )
        #
        np.copyto(prevZ, Z)
        DY = np.dot(Y,D.T)
        DYpU = DY + U
        prox_g(DYpU, uzawa_mu, args,  Z)
        #
        # U(k+1) <- U(k) + Y(k+1) - Z(k+1)
        #
        dU = DY - Z
        if kappa < 1:
            tau *= kappa
            uzawa_mu *= kappa
            U += (1/kappa)*dU
        else:
            U += dU
        #
        # convergence
        #
        ndU = np.linalg.norm(dU,'fro')/np.linalg.norm(U,'fro')       
        dX = np.linalg.norm(Y - prevY, 'fro') / (1e-10 + np.linalg.norm(Y, 'fro'))
        dW = np.linalg.norm(Z - prevZ, 'fro') / (1e-10 + np.linalg.norm(Z, 'fro'))

        if not iter % 10:
            Xerr = np.linalg.norm(Y - Xtrue, 'fro') / np.linalg.norm(Xtrue, 'fro')
            merr = np.linalg.norm(np.dot(Y, P.T) - B, 'fro') / np.linalg.norm(B)
            patmap.stitch(Y, width, stride, nrows, ncols, Irec)
            Irec = np.minimum(1.0,np.maximum(0.0,Irec))
            Ierr = np.sqrt(np.mean((Irec-Itrue)**2))
            #SSIM = ssim.ssim(Itrue.astype(np.uint32),Irec.astype(np.uint32))
            psnr = 20*np.log10(1.0/Ierr)
            #print(f"tau={tau:8.5f} dX={dX:8.5f} dW={dW:8.5f} dU={ndU:8.5f} Xerr={Xerr:8.5f} merr={merr:8.5f} IMAGE: MSE={Ierr:8.5f} PSNR={psnr:8.5f} SSIM={SSIM:8.6f}")
            print(f"iter={iter:05d} tau={tau:8.5f} dX={dX:8.5f} dW={dW:8.5f} dU={ndU:8.5f} Xerr={Xerr:8.5f} merr={merr:8.5f} IMAGE: MSE={Ierr:8.5f} PSNR={psnr:8.5f}")
            plt.imsave(f"iter{iter:04d}.png",Irec,cmap=cm.gray)
            if dX < args.outer_tol and dW < args.outer_tol:
                print('converged to tolerance.')
                break
    #
    # GUARDAR SALIDA
    #
    plt.imsave(args.output,Irec,cmap=cm.gray)
#==============================================================================
