#!/bin/bash
# parameters similar to those used by Mon and Fowler in their paper
# 
./paco_bcs.py -w 32 -s 30 --proj-num 180 --tau 0.1 --inner-tau 0.1 --maxiter 100 --inner-tol 7e-4 --proj-type binary --diff-type bi lena.png lena_bcs.png $*

# subsampling 22% 32.30dB
#./paco_bcs.py -w 32 -s 30 --proj-num 200 --tau 0.1 --inner-tau 0.1 --maxiter 100 --inner-tol 7e-4 --proj-type binary --diff-type bi lena.png lena_bcs.png $*
