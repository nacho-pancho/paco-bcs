#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Simple patch extraction and stitching in Python
#
import numpy as np

def grid_size(l,w,s):
    return int(np.ceil((l-w)/s) + 1)


def padded_size(l,w,s):
    g = grid_size(l,w,s)
    return (g-1)*s + w


def pad_image(img, w, s):
    '''
    Enlarge image so that it is exactly covered by the tiling defined by
    the patch width w and stride s
    '''
    d0,d1 = img.shape
    dd0 = padded_size(d0,w,s)
    dd1 = padded_size(d1,w,s)
    padded_img = np.zeros((dd0,dd1))
    padded_img[ :d0, :d1 ] = img
    # mirror is best for DCT
    if dd0 > d0:
        pad0 = dd0-d0
        padded_img[ d0:, :d1 ] = img[ (d0-1):(d0-1-pad0):-1, : ]
    if dd1 > d1:
        pad1 = dd1-d1
        padded_img[ :d0, d1: ] = img[ :, (d1-1):(d1-1-pad1):-1 ]
    if dd0 > d0 and dd1 > d1:
        padded_img[ d0:, d1: ] = img[ (d0-1):(d0-1-pad0):-1, (d1-1):(d1-1-pad1):-1 ]
    return padded_img


def extract(img, w, s, x = None):  # s = stride
    M,N = img.shape
    gm = grid_size(M,w,s)
    gn = grid_size(N,w,s)
    n = gm*gn
    m = w*w
    if x is None:
        x = np.zeros((n,m))
    for i in range(gm):
        for j in range(gn):
            x[i * gn + j, :] = img[ (i * s):(i * s + w), (j * s):(j * s + w) ].ravel()
    return x

def extract_cmaj(img, w, s, x = None):  # s = stride
    '''
    extract patches in column-major ordering (concatenate patch columns instead of rows)
    :param img:
    :param w:
    :param s:
    :param x:
    :return:
    '''
    M,N = img.shape
    gm = grid_size(M,w,s)
    gn = grid_size(N,w,s)
    n = gm*gn
    m = w*w
    if x is None:
        x = np.zeros((n,m))
    for i in range(gm):
        for j in range(gn):
            x[i * gn + j, :] = img[ (i * s):(i * s + w), (j * s):(j * s + w) ].T.ravel()
    return x

def build_normalization_matrix(img,w,s):
    M, N = img.shape
    gm = grid_size(M,w,s)
    gn = grid_size(N,w,s)
    nimg = np.zeros(img.shape)
    for i in range(gm):
        for j in range(gn):
            nimg[(i * s):(i * s + w), (j * s):(j * s + w)] = nimg[(i * s):(i * s + w), (j * s):(j * s + w)] + 1
    return 1.0 / nimg

norm_img = None

def stitch(x, w, s, M, N, img = None):
    global norm_img
    if img is None:
        img = np.empty((M,N))

    if norm_img is None:
        norm_img = build_normalization_matrix(img,w,s)

    gm = grid_size(M,w,s)
    gn = grid_size(N,w,s)
    n = gm*gn
    m = w*w
    img[:] = 0
    for i in range(gm):
        for j in range(gn):
            patch = np.reshape(x[i * gn + j, :], (w, w))
            img[(i * s):(i * s + w), (j * s):(j * s + w) ] = img[(i * s):(i * s + w), (j * s):(j * s + w) ] + patch
    img[:] = img * norm_img
    return img

def stitch_cmaj(x, w, s, M, N, img = None):
    global norm_img
    if img is None:
        img = np.empty((M,N))

    if norm_img is None:
        norm_img = build_normalization_matrix(img,w,s)

    gm = grid_size(M,w,s)
    gn = grid_size(N,w,s)
    n = gm*gn
    m = w*w
    img[:] = 0
    for i in range(gm):
        for j in range(gn):
            xij = x[i * gn + j, :]
            #patch = np.reshape(xij, (w, w))
            patchT = np.reshape(xij, (w, w), order='F')
            img[(i * s):(i * s + w), (j * s):(j * s + w) ] = img[(i * s):(i * s + w), (j * s):(j * s + w) ] + patchT
    img[:] = img * norm_img
    return img

