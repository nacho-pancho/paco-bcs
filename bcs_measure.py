#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#==============================================================================

"""
PACO Block Compressed Sensing
Utility for computing compressed measurements
Input: 
- image
- patch width
- stride
- number of compressed samples
- random seed
- measurement matrix
Output:
- geometric information: first row: nrows, ncols
                         following rows: patch upper-left coordinates
- compressed measurements

@author: nacho
"""
import os
import sys
import skimage.io as imgio
import numpy as np
from numpy.random import default_rng
import argparse
import patch_mapping as patmap
import operators

#==============================================================================

if __name__ == '__main__':
    epilog = "Output image file name is built from input name and parameters."
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--stride", type=int, required=True,
                        help="patch stride")
    parser.add_argument("-w", "--width", type=int, required=True,
                        help="patch width")
    parser.add_argument("-D","--dtype", type=str, default="bi",
                        help="Type of differential operator.")
    parser.add_argument("-t","--type", type=str, default="random",
                        help="Kind of projection matrix (random or binary).")
    parser.add_argument("-k","--samples", type=int,required=True,
                        help="Number of samples.")
    parser.add_argument("-r","--seed", type=int,required=True,
                        help="Random seed (integer).")
    parser.add_argument("-i","--input", help="input image file")
    parser.add_argument("-o","--outdir", help="output directory")
    args = parser.parse_args()

    cmd = " ".join(sys.argv)
    print(("Command: " + cmd))
    #
    #
    # parametros (por ahora mejores para desfile1 por lo menos)
    #
    # buenos parametros: 4x8x8+1+2+#
    width  = args.width
    stride = args.stride
    k      = args.samples
    seed   = args.seed
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir,exist_ok=True)

    P = operators.create_proj_op(width*width,k,type=args.type,seed=seed)

    I = imgio.imread(args.input)*(1/255)
    I = patmap.pad_image(I,width,stride)
    
    nrows,ncols = I.shape
    npixels = nrows*ncols
    X = patmap.extract(I,width,stride)
    n,m = X.shape

    B = np.dot(X,P.T)
    nsamples = n*k
    cratio   = n*k / npixels
    print("dim =",m,"blocks =",n," samples =",nsamples, " cratio =",cratio)
    path_name, base_name = os.path.split(args.input)
    base_name, file_ext = os.path.splitext(base_name)
    #base_name = args.input[:-4] # strip dot and extension and add
    suffix = f'w{width}_{args.type}_k{k}_r{seed}.txt'
    
    out_fname = os.path.join(args.outdir,f'{base_name}_samples_s{stride}_{suffix}')
    np.savetxt(out_fname,B,'%10f')
    # not used, but useful to have around
    diff_file = os.path.join(args.outdir,f'D_w{width}.txt')
    if True: #not os.path.exists(diff_file):
        print(args.dtype)
        D = operators.create_diff_op(width,width,type=args.dtype)
        np.savetxt(diff_file,D,fmt='%8.5f')

    proj_file = os.path.join(args.outdir,f'P_{suffix}')
    if True: #not os.path.exists(proj_file):
        np.savetxt(proj_file,P,fmt='%8.5f')




#==============================================================================
