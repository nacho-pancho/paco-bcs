#!/bin/bash
w=10

# good combination
s=9
k=27 #30%
t=0.1 

# testing combination
s=8
k=20 #30%
t=0.01
ka=0.995

r=1863699
r=8636991
I=lena
outdir="results/lena_dict_w${w}_s${s}_random_k${k}_r${r}"
./code/bcs_measure.py -D bi -o ${outdir} -w ${w} -k ${k} -s ${s} -r ${r}  -i data/${I}.png
./code/bcs_paco_dict.py --save-diag \
	-w ${w} -s ${s} \
	--tau ${t} --mu 0.95 --kappa ${ka} --maxiter 500 \
	--diff-op data/dict-w10.txt \
	--meas-op ${outdir}/P_w${w}_random_k${k}_r${r}.txt \
	--samples  ${outdir}/${I}_samples_s${s}_w${w}_random_k${k}_r${r}.txt \
	--save-iter\
	--reference data/${I}.png \
	--outdir ${outdir} $*
ffmpeg -i ${outdir}/iter%03d0.png -framerate 5 -vcodec copy ${outdir}/${I}_w${w}_s${s}_random_k${k}_s${r}.mkv
#rm ${outdir}/iter*png
