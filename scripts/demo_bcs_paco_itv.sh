#!/bin/bash
w=32
s=24
#k=58 10%
k=288 #50%
r=1863699
r=8636991
I=lena
outdir="results/lena_w${w}_s${s}_random_k${k}_r${r}"
./code/bcs_measure.py -D bi -o ${outdir} -w ${w} -k ${k} -s ${s} -r ${r} $* -i data/${I}.png
./code/bcs_paco_itv.py --save-diag \
	-w 32 -s 24 \
	--tau 10 --mu 0.95 --maxiter 200 \
	--diff-op ${outdir}/D_w${w}.txt \
	--meas-op ${outdir}/P_w${w}_random_k${k}_r${r}.txt \
	--samples  ${outdir}/${I}_samples_s${s}_w${w}_random_k${k}_r${r}.txt \
	--reference data/${I}.png \
	--outdir ${outdir} $*
ffmpeg -i ${outdir}/iter%03d0.png -framerate 5 -vcodec copy ${outdir}/${I}_w${w}_s${s}_random_k${k}_s${r}.mkv
rm ${outdir}/iter*png
