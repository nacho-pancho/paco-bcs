#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#==============================================================================

"""
PACO Block Compressed Sensing / DCT

This version minimizes the L1 norm of the DCT coefficients of the patches
instead of their Total Variation.

@author: nacho
"""
import os
import numpy as np
from numpy.random import default_rng
import argparse
import patch_mapping as patmap
import time
import skimage.io as imgio
from scipy import fft

rng = default_rng(42)
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

def prox_g(x, tau, px):
    """
    :param x: proximal function argument
    :param px: output
    :param args: program args
    """
    w = int(np.sqrt(px.shape[1]))
    if px is None:
        px = np.empty(x.shape)
    for i in range(x.shape[0]):
        patch = np.reshape(x[i, :], (w, w))
        fft.dctn(patch,norm="ortho", overwrite_x=True)
        patch[np.abs(patch) < tau] = 0
        patch[patch > tau] -=  tau # without these two lines it would be hard thresholding
        patch[patch < -tau] += tau #
        fft.idctn(patch,norm="ortho", overwrite_x=True)
        px[i, :] = patch.ravel()
    return px


#==============================================================================

PtPPti = None
ImPtPPtiP = None
PtPPtiB = None
POCSimg = None
def prox_f(YpU,P,B,args,Z):
    """
    POCS for projecting onto the intersection of the consensus set and the CS set
    Z <- Z - Pt(PPt)i(PZ - B)
    written in transposed form (B = ZP, BPt = ZPPt, etc...)
    """
    global PtPPti, ImPtPPtiP, POCSimg, PtPPtiB
    if ImPtPPtiP is None:
        PPt = np.dot(P,P.T)
        PPti = np.linalg.inv(PPt)
        PtPPti = np.dot(P.T,PPti)
        PtPPtiB = np.dot(B,PtPPti.T)
        ImPtPPtiP = np.dot(PtPPti,P)
        ImPtPPtiP = np.eye(m) - ImPtPPtiP
        if args.save_diag:
            np.savetxt(os.path.join(args.outdir,'PPt_py.txt'),PPt,fmt="%8.6f")
            np.savetxt(os.path.join('PPti_py.txt'),PPti,fmt="%8.6f")
            np.savetxt(os.path.join(args.outdir,'PtPPti_py.txt'),PtPPti,fmt="%8.6f")
            np.savetxt(os.path.join(args.outdir,'ImPtPPtiP_py.txt'),ImPtPPtiP,fmt="%8.6f")
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
        Z2 = np.dot(Z1,ImPtPPtiP.T) + PtPPtiB 
        #
        # convergence metrics
        #
        dZ2 = np.linalg.norm(Z2prev-Z2,'fro')/ np.prod(Z2.shape)
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
    parser.add_argument("--mu", type=float, default=0.99,
                        help="uzawas mu")
    parser.add_argument("-e", "--eps", type=float, default=DEF_WEPS,
                        help="Smoothing constant in L1 weights estimation.")
    parser.add_argument("-P","--meas-op", type=str, required=True,
                        help="Measurement matrix.")
    parser.add_argument("-B","--samples", type=str, required=True,
                        help="Samples matrix.")
    parser.add_argument("-D","--diff-op", type=str, required=True,
                        help="Differential operator for TV norm.")
    parser.add_argument("--inner-tol", type=float, default=1e-4,
                        help="Tolerance for inner ADMM.")
    parser.add_argument("--outer-tol", type=float, default=1e-4,
                        help="Tolerance for outer ADMM.")
    parser.add_argument("--save-iter", action="store_true",
                        help="If specified, save intermediate images.")
    parser.add_argument("--save-diag", action="store_true",
                        help="If specified, save precomputed matrices and other diagnostics.")
    parser.add_argument("--inner-maxiter", type=int, default=100,
                        help="Maximum number of iterations in inner iter.")
    parser.add_argument("--maxiter", type=int, default=100,
                        help="Maximum number of iterations in ADMM.")
    parser.add_argument("--reference", type=str, required=True, help="reference image file")
    parser.add_argument("--outdir", type=str, default=".", help="output directory")
    args = parser.parse_args()

    #cmd = " ".join(sys.argv)
    print('Arguments:')
    dargs = vars(args)
    for k in dargs.keys():
        v = dargs[k]
        print(f'\t{k:8}:{v:8}')
    #print(args)
    #
    #
    # parametros (por ahora mejores para desfile1 por lo menos)
    #
    # buenos parametros: 4x8x8+1+2+#
    width  = args.width
    stride = args.stride

    Itrue = imgio.imread(args.reference)
    nrows0,ncols0 = Itrue.shape
    Itrue = patmap.pad_image(Itrue,width,stride)
    
    nrows,ncols = Itrue.shape
    npixels = nrows*ncols

    Xtrue = patmap.extract(Itrue,width,stride)

    n,m = Xtrue.shape
    P = np.loadtxt(args.meas_op)
    B = np.loadtxt(args.samples)
    if B.shape[0] != n:
        print(f'ERROR: number of compressed signals {B.shape[0]} should be {n}.\n')
        exit(1)
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
    if args.save_diag:
        np.savetxt(os.path.join(args.outdir,'PtP_py.txt'),PtP,fmt='%8.6f')
        np.savetxt(os.path.join(args.outdir,'PtPi_py.txt'),PtPi,fmt='%8.6f')
        np.savetxt(os.path.join(args.outdir,'PtPiPt_py.txt'),PtPiPt,fmt='%8.6f')
        np.savetxt(os.path.join(args.outdir,'Y0_py.txt'),Y,fmt='%8.6f')

    prevY = np.zeros(Y.shape)   # and its previous value
    Z = np.zeros((n,m)) #
    prevZ = np.zeros(Z.shape)   # previous value
    U = np.zeros(Z.shape)       # ADMM multipliers
    tau = args.tau              # ADMM penalty
    maxiter = args.maxiter      # maximum number of iterations
    Irec = np.empty(Itrue.shape) # reconstructed image
    #
    # main Linearized ADMM loop
    #
    # min f(Y) + g(DY)
    # min f(Y) + g(Z) s.t. DY=Z
    #
    print('tau',args.tau, 'mu',args.mu)
    nm = n * m
    tau0 = args.tau
    cost = 0
    prev_cost = 0
    dcost = 1
    stats = list()
    t0 = time.time()
    tau = tau0
    tau_prev = tau
    for iter in range(maxiter):
        #
        # Y(k+1) <- prox_{tf}( Z(k) - U(k) )
        #
        np.copyto(prevY, Y)
        #prox_f(Y - (mu / tau) * np.dot(DY - (Z - U), D) , P, B, args, Y)
        prox_f(Z - U, P, B, args, Y)
        #
        # Z(k+1) <- prox_{tg}( Y(k+1) + U(k) )
        #
        np.copyto(prevZ, Z)
        prox_g(Y + U, tau, Z)
        #
        # U(k+1) <- U(k) + Y(k+1) - Z(k+1)
        #
        dU = Y - Z
        tau_prev = tau
        kappa = tau/tau_prev
        U += (1/kappa)*dU
        #
        # see if we increase or decrese stepsize
        #
        fact = 0.995 # fact = 0.99


        #tau = tau0 / np.power(iter+1,0.5)
        tau *= fact
        #
        # convergence
        #
        prev_cost = cost
        ndU = np.linalg.norm(dU,'fro')/(1e-10+np.linalg.norm(U,'fro'))
        cost = (np.sum(np.abs(Y)) + np.dot(dU.ravel(),U.ravel()) + (0.5/tau)*ndU**2)/np.prod(dU.shape)
        dcost = prev_cost - cost
        if not iter % 10:
            dX = np.linalg.norm(Y - prevY, 'fro') / (1e-10 + np.linalg.norm(Y, 'fro'))
            dZ = np.linalg.norm(Z - prevZ, 'fro') / (1e-10 + np.linalg.norm(Z, 'fro'))
            E    = np.dot(Y, P.T)-B
            nE   = np.linalg.norm(E,'fro')
            merr = nE / np.linalg.norm(B)
            patmap.stitch(Y, width, stride, nrows, ncols, Irec)
            Irec = np.minimum(1.0,np.maximum(0.0,Irec))
            Ierr = np.sqrt(np.mean((Irec-Itrue)**2))
            psnr = 20*np.log10(1.0/Ierr)
            dt = (time.time() - t0)
            print(f"iter={iter:05d} tau={tau:8.5f} dX={dX:8.5f} dZ={dZ:8.5f} dU={ndU:8.5f} dcost={dcost:8.5f} merr={merr:8.5f} IMAGE: MSE={Ierr:8.5f} PSNR={psnr:8.5f} dt={dt:8.1f}s")
            stats.append([iter,tau,dX,dZ,ndU,cost,merr,Ierr,psnr])
            if args.save_iter:
                imgio.imsave(os.path.join(args.outdir,f"iter{iter:04d}.png"),np.round(255*Irec).astype(np.uint8))
            if dX < args.outer_tol and dZ < args.outer_tol:
                print('converged to tolerance.')
                break
    #
    # GUARDAR SALIDA
    Irec = Irec[:nrows0,:ncols0]
    statmat = np.array(stats,dtype=object)
    np.savetxt(os.path.join(args.outdir,"optimization.txt"),statmat,fmt='%8.6f')
    imgio.imsave(os.path.join(args.outdir,'recovered.png'),np.round(255*Irec).astype(np.uint8))

#==============================================================================
