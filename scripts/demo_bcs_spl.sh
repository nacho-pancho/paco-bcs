#!/bin/bash
./bcs_spl.py -w 16 --max-iter 100 --proj-type binary --proj-num 50  lena.png lena_spl.png $*
