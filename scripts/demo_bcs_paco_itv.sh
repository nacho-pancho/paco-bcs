#!/bin/bash
w=32
#s=24
s=31 #1 pixel overlap, 
k=288 #30%
r=1863699
r=8636991
I=lena
outdir="results/lena_itv_w${w}_s${s}_random_k${k}_r${r}"
./code/bcs_measure.py -D bi -o ${outdir} -w ${w} -k ${k} -s ${s} -r ${r} $* -i data/${I}.png
./code/bcs_paco_itv.py --save-diag \
	-w ${w} -s ${s} \
	--tau 10 --mu 0.95 --maxiter 500 \
	--diff-op ${outdir}/D_w${w}.txt \
	--meas-op ${outdir}/P_w${w}_random_k${k}_r${r}.txt \
	--samples  ${outdir}/${I}_samples_s${s}_w${w}_random_k${k}_r${r}.txt \
	--reference data/${I}.png \
	--outdir ${outdir} $*
ffmpeg -i ${outdir}/iter%03d0.png -framerate 5 -vcodec copy ${outdir}/${I}_w${w}_s${s}_random_k${k}_s${r}.mkv
rm ${outdir}/iter*png
