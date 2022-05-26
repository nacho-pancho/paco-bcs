#!/bin/bash
w=16
s=8
k=13
r=1863699
I=lena
outdir="results/lena_w${w}_s${s}_random_k${k}_r${r}"
./code-bcs/bcs_measure.py -D bi -o ${outdir} -w ${w} -k ${k} -s ${s} -r ${r} -i data/image/classic/${I}.png
python3 -m cProfile -s cumulative ./code-bcs/bcs_paco_itv.py --save-diag \
	-w 16 -s 8 \
	--tau 10 --mu 0.95 --maxiter 100 \
	--diff-op ${outdir}/D_w${w}.txt \
	--meas-op ${outdir}/P_w${w}_random_k${k}_r${r}.txt \
	--samples  ${outdir}/${I}_samples_s${s}_w${w}_random_k${k}_r${r}.txt \
	--reference data/image/classic/${I}.png \
	--outdir ${outdir} $*
ffmpeg -i ${outdir}/iter%03d0.png -framerate 5 -vcodec copy ${outdir}/${I}_w${w}_s${s}_random_k${k}_s${r}.mkv
rm ${outdir}/iter*png
