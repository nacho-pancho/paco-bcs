#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#==============================================================================

"""
PACO Block Compressed Sensing
@author: nacho
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import patch_mapping as patmap
import paco.linop as linop
import operators


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

#==============================================================================


def cs_itv_prox_solve_prox_g(z,inner_tau):
    '''
    Solves

    v = arg min_v \sum_{i=1}^n ||(v_{2i-1},v_{2i}||_2 + 1/2tau ||v-z||_2^2

    This is a vector thresholding applied to all groups of 2 elements in z
    :param z: input vector to be thresholded; the form is (dx1,dy1,dx2,dy2,...)
    :returns v: the thresholded version of z
    '''
    N2 = len(z)
    N = int(N2/2)
    z = np.reshape(z,(2,N))
    nz = np.sqrt(np.sum(z**2,axis=0))
    tz = np.outer(np.ones((2,1)),np.maximum(0,1-inner_tau/(nz+1e-10)))
    return np.reshape(tz*z,(2*N))

#==============================================================================

Ca = None
Cb = None
prev_tau = -1
prev_taup = -1

def cs_itv_prox_solve_prox_f(D,z,w,P,b,tau,taup):
    '''
    Solve proximal operator of proximal operator of f.
    Yes, sounds ugly.

    Solve arg (1/2tau) ||y - w||_2^2 + (1/2tau')||Dy - z||_2^2 s.t. Py=b

    :param D: differential operator
    :param z: proximal operator argument
    :param w: proximal point from outer ADMM
    :param P: compressed sensing matrix
    :param b: compressed samples
    :param tau: ADMM parameter of OUTER ADMM
    :param taup: ADMM parameter of THIS ADMM
    :returns y: the solution
    '''
    #
    # y = Gi [ a - Pt * (P*Gi*Pt)i * (P*Gi*a - b) ]
    #
    # see paper for derivations
    # the names of the variables are the same with the exception of:
    #
    # tau = outer_tau
    # tau' = inner_tau
    #
    # we rewrite these so that we don't need to re-compute
    # matrices and their inverses:
    #
    # y = Gi (I - Pt * (P*Gi*Pt)i * P* Gi) *a - Gi * Pt * (P*Gi*Pt)i * b
    # y = [Gi - (P*Gi)t * (P*Gi*Pt)i * P * Gi ]*a + [(P*Gi)t * (P*Gi*Pt)i] * b
    # y = A a - B*b
    #
    # where
    #
    # A = [ Gi - (P*Gi)t * (P*Gi*Pt)i * P * Gi ]
    # B = [ (P*Gi)t * (P*Gi*Pt)i ]
    # a = [ (1/tau)*w + (1/tau')*Dt*z ]
    #
    global prev_tau
    global prev_taup
    global Ca,Cb
    a = (1 / tau) * w + (1 / taup) * np.dot(D.T, z)
    if tau != prev_tau or prev_taup != taup:
        prev_tau = tau
        prev_taup = taup
        G     = np.dot(D.T,D)
        Gi    = np.linalg.inv( (1/taup)*G + (1/tau)*np.eye(G.shape[0]) )
        PGi   = np.dot(P,Gi)
        PGiPti =  np.linalg.inv(np.dot(PGi,P.T))
        Ca = Gi - np.dot(PGi.T,np.dot(PGiPti,PGi))
        Cb = np.dot(PGi.T,PGiPti)
    y = np.dot(Ca, a) + np.dot(Cb, b)
    return y

#==============================================================================

def cs_itv_prox(D,P,b,w,args, y0=None):
    '''
    Proximal operator of the Compressive Sensing/Isotropic Total Variation problem.
    Solves
    y = arg min ||Dy||_TV + (1/2tau)||y-w||
        s.t. Py = b
    :param D: differential operator
    :param P: compressed sensing matrix
    :param b: compressed samples
    :param w: proximal argument
    :param y0: initial iterate
    :param outer_tau: penalty of the outer ADMM
    :returns y: the solution

    '''
    outer_tau = args.tau
    inner_tau = args.inner_tau
    tol = args.inner_tol
    maxiter = args.inner_maxiter
    m = D.shape[1]
    if y0 is None:
        y = np.zeros(m)
    else:
        y = np.copy(y0)
    t = np.zeros(2*m)
    prevy = np.zeros(m)
    for iter in range(maxiter):
        np.copyto(prevy,y)
        v = cs_itv_prox_solve_prox_g( D.dot(y) - t, inner_tau )
        y = cs_itv_prox_solve_prox_f( D, v+t, w, P, b, outer_tau, inner_tau )
        t += v - D.dot(y)

        dif = np.linalg.norm(y-prevy)/(1e-10+np.linalg.norm(y))
        if dif < tol:
            break
    if dif > tol:
        print(f'inner ADMM did not converge to tolerance ({dif}>{tol}) after {maxiter} iterations.')
    return y

#==============================================================================

def cs_bp_prox(D,P,b,w,args, y0=None):
    return None

#==============================================================================

def cs_spl_prox(D,P,b,w,args, y0=None):
    '''
    Smoothed Landweber Projection
    '''
    return None

# ==============================================================================

def solve_prox_f(D,P,B,W,args,Y0=None,type='itv'):
    '''
    :param D: differential operator on x (for computing TV)
    :param P: compressed sensing matrix
    :param B: compressed samples
    :param tau: ADMM penalty of the *outer* problem
    :
    '''
    if type == 'itv':
        prox_f = cs_itv_prox # TV CS; D is a differential operator
    elif type == 'bp':
        prox_f = cs_bp_prox  # traditional Basis Pursuit; D is the sparsifying transform
    elif type == 'ht':
        prox_f = cs_spl_prox  # Smoothed Projected Landweber
    m = D.shape[1]
    n = B.shape[0]   
    Y = np.empty((n,m))
    for j in range(n):
        bj  = B[j,:]
        wj  = W[j,:]
        y0j = Y0[j,:]
        Y[j,:] = prox_f(D, P, bj, wj, args, y0j)
    return Y

#==============================================================================

def solve_prox_g(Z,tau):
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
    parser.add_argument("-s", "--stride", type=int, default=DEF_STRIDE,
                        help="patch stride")
    parser.add_argument("-w", "--width", type=int, default=DEF_WIDTH,
                        help="patch stride")
    parser.add_argument("-t", "--tau", type=float, default=DEF_ADMM_PENALTY,
                        help="ADMM penalty")
    parser.add_argument("-i", "--inner-tau", type=float, default=0.5,
                        help="ADMM penalty")
    parser.add_argument("-k", "--kappa", type=float, default=0.95,
                        help="Multiplier for diminishing ADMM stepsize.")
    parser.add_argument("-mi", "--maxiter", type=int, default=DEF_MAX_ITER,
                        help="Maximum ADMM iterations")
    parser.add_argument("-e", "--eps", type=float, default=DEF_WEPS,
                        help="Smoothing constant in L1 weights estimation.")
    parser.add_argument("-mc", "--minchange", type=float, default=DEF_MIN_CHANGE,
                        help="Minimum change between ADMM iterations")
    parser.add_argument("-mm", "--mmap", type=bool, default=DEF_USE_MMAP,
                        help="Use MMAP for storage. Set to yes if physical memory is not enough.")
    parser.add_argument("-rw", "--rwiter", type=int, default=DEF_REWEIGHT_ITER,
                        help="How many L1 reweighting iterations to apply.")
    parser.add_argument("--proj-num", type=int, default=0,
                        help="Number of compressed sensing projections.")
    parser.add_argument("--proj-type", type=str, default="random",
                        help="Type of compressed sensing operator (random, binary, dct, idct).")
    parser.add_argument("--diff-type", type=str, default="uni",
                        help="Type of differential  operator (uni,bi).")
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
    stride = args.stride

    Itrue = plt.imread(args.input)
    Itrue = patmap.pad_image(Itrue,width,stride)
    
    nrows,ncols = Itrue.shape
    npixels = nrows*ncols

    Xtrue = patmap.extract(Itrue,width,stride)

    n,m = Xtrue.shape
    print("Signal dimension =",m)
    print("Number of blocks =",n)
    P = operators.create_proj_op(m,args.proj_num,args.proj_type,seed=42)
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
    # B   = X*Pt
    # B*P = X*PtP
    # B*P(PtP)' = X
    PPtPi = np.dot(P,np.linalg.inv(np.dot(P.T, P)+10*np.eye(P.shape[1]))) # a little Ridge reg.
    Y = np.dot(B,PPtPi) # first approximation using pseudoinverse


    prevY = np.zeros(Y.shape)   # and its previous value
    Z = np.zeros(Xtrue.shape)   # SECOND ADMM variable
    prevZ = np.zeros(Z.shape)   # previous value
    U = np.zeros(Z.shape)       # ADMM multipliers
    tau = args.tau              # ADMM penalty
    maxiter = args.maxiter      # maximum number of iterations
    kappa = float(args.kappa)   # penalty increase factor
    Irec = np.empty(Itrue.shape) # reconstructed image
    #
    # main ADMM loop
    #
    for iter in range(maxiter):
        print(f"iter={iter:05}",end=", ")
        #
        # Y(k+1) <- prox_{tf}( Z(k) - U(k) )
        #
        np.copyto(prevY, Y)
        Y[:] = solve_prox_f(D, P, B, Z - U,args, Y0=prevY, type='itv')
        #
        # Z(k+1) <- prox_{tg}( Y(k+1) + U(k) )
        #
        np.copyto(prevZ, Z)
        Z[:] = solve_prox_g(Y + U, tau)
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

        Xerr = np.linalg.norm(Y - Xtrue, 'fro') / np.linalg.norm(Xtrue, 'fro')
        merr = np.linalg.norm(np.dot(Y, P.T) - B, 'fro') / np.linalg.norm(B)
        Irec = patmap.stitch(Y, width, stride, nrows, ncols, Irec)
        Ierr = np.sqrt(np.mean((Irec-Itrue)**2))
        psnr = 20*np.log10(1.0/Ierr)
        print(f"tau={tau:8.5f} dX={dX:8.5f} dW={dW:8.5f} dU={ndU:8.5f} Xerr={Xerr:8.5f} merr={merr:8.5f} IMAGE: MSE={Ierr:8.5f} PSNR={psnr:8.5f}")
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
