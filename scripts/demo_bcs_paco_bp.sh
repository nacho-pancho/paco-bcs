#!/bin/bash
./bcs_paco_bp.py -w 16 -s 12 --tau 0.1 --maxiter 100 --inner-tol 1e-3 --proj-type random --proj-num 32 --diff-type bi lena.png lena_bcs_bp.png $*
