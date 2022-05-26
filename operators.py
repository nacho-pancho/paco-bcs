#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#==============================================================================

"""
PACO Block Compressed Sensing
@author: nacho
"""
import numpy as np
from numpy.random import default_rng
import sys


#==============================================================================

def create_diff_op(nrows,ncols,type='bi'):
    '''
    Computes the matrix form of a differential operator on 2D images, for example, the Laplacian
    '''
    N = nrows*ncols
    if type == 'uni':
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

    if type == 'uni2':
        Dh = np.eye(N,N)
        li = 0
        for i in range(nrows):
            for j in range(ncols):
                if j > 0:
                    Dh[li,i*ncols+(j-1)] = -0.5
                    Dh[li,i*ncols+(j  )] =  0.5
                li = li+1
        Dv = np.eye(N,N)
        li = 0
        for i in range(nrows):
            for j in range(ncols):
                if i > 0:
                    Dv[li,(i-1)*ncols+j] = -0.5
                    Dv[li,(i  )*ncols+j] =  0.5
                li = li+1

    elif type == 'bi2':
        Dh = np.zeros((N,N))
        li = 0
        for i in range(nrows):
            Dh[li,i*ncols] = 1
            li = li + 1
            for j in range(1,ncols-1):
                Dh[li,i*ncols+(j-1)] = -0.5
                Dh[li,i*ncols+(j+1)] =  0.5
                li = li + 1
            Dh[li,i*ncols+(ncols-1)] = 1
            li = li+1
        Dv = np.zeros((N,N))
        li = 0
        for j in range(ncols):
            Dv[li,j] = 1
            li = li + 1
            for i in range(1,nrows-1):
                Dv[li,(i-1)*ncols+j] = -0.5
                Dv[li,(i+1)*ncols+j] =  0.5
                li = li+1
            Dv[li,(nrows-1)*ncols+j] =  1
            li = li+1

    elif type == 'bi':
        Dh = np.eye(N,N)
        li = 0
        for i in range(nrows):
            for j in range(ncols):
                v = 0
                if j > 0:
                    v += 1
                    Dh[li,i*ncols+(j-1)] = -1
                if j < (ncols-1):
                    v += 1
                    Dh[li,i*ncols+(j+1)] = -1
                if v > 1:
                    Dh[li,li] = v
                li = li+1
        Dv = np.eye(N,N)
        li = 0
        for i in range(nrows):
            for j in range(ncols):
                v = 0
                if i > 0:
                    Dv[li,(i-1)*ncols+j] = -1
                    v += 1
                if i < (nrows-1):
                    Dv[li,(i+1)*ncols+j] = -1
                    v += 1
                if v > 1:
                    Dv[li,li] = v
                li = li+1
    elif type == 'laplacian':
        # PENDING
        Dh = np.eye(N,N)
        li = 0
        for i in range(nrows):
            for j in range(ncols):
                v = 0
                if j > 0:
                    v += 1
                    Dh[li,i*ncols+(j-1)] = -1
                if j < (ncols-1):
                    v += 1
                    Dh[li,i*ncols+(j+1)] = -1
                if v > 1:
                    Dh[li,li] = v
                li = li+1
        Dv = np.eye(N,N)
        li = 0
        for i in range(nrows):
            for j in range(ncols):
                v = 0
                if i > 0:
                    Dv[li,(i-1)*ncols+j] = -1
                    v += 1
                if i < (nrows-1):
                    Dv[li,(i+1)*ncols+j] = -1
                    v += 1
                if v > 1:
                    Dv[li,li] = v
                li = li+1
    else:
        return None
    D = np.empty((2*N,N))
    D[0::2,:] = Dh
    D[1::2,:] = Dv
    return D
 
#==============================================================================

def create_proj_op(m,K,type='random',seed=18636998):
    rng = default_rng(seed)
    if type == 'random' or type == 'binary':
        Wn = (1.0/np.sqrt(m))*rng.normal(size=(m,m))
        # EXPERIMENTAL: add DC term
        #Wn[0,:] = 1.0/np.sqrt(m)
        Wn[0,:] = 1.0/m
        Pk = Wn[:K,:]
        _,_,P = np.linalg.svd(Pk,full_matrices=False)
        #P = Pk
        if type == 'binary':
            P = 1-2*(P>0)
            nP = np.sum(P**2,axis=1)
            nP[0] = 1 # do not scale DC term
            W = np.diag(1.0/np.sqrt(nP))
            P = np.dot(W,P)
        return P
    else:
        return None


#==============================================================================

def create_sparse_op(nrows,ncols,type='dct'):
    '''
    Computes the matrix form of the sparsifying operator
    '''
    n = nrows*ncols
    W = np.eye(n)
    D = np.copy(W)
    linop.dct2d(W,nrows,ncols,D)
    nD = np.sum(D**2,axis=1)
    W = np.diag(1.0 / np.sqrt(nD))
    D = np.dot(W, D)
    return D

if __name__ == '__main__':
    #
    # when called with two arguments,
    # creates projection matrices and stores them
    #
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} patch_width num_samples random_seed')
        exit(1)

    w = int(sys.argv[1])
    k = int(sys.argv[2])
    s = int(sys.argv[3])

    m = w*w
    D = create_diff_op(w,w)
    P = create_proj_op(m,k)
    np.savetxt(f'diff_op_w{w:03d}_p{k:03}_s{s:010d}.txt',D,fmt='%8.5f')
    np.savetxt(f'proj_op_w{w:03d}_p{k:03}_s{s:010d}.txt',P,fmt='%8.5f')

