#!/bin/bash
w=10
s=9
r=8636991
I=lena
img=data/image/gray/classic/${I}.png
./code/bcs_measure.py -D bi -o ${outdir} -w ${w} -r ${r} -s ${s} -S ${S} $* -i ${img}
./code/bcs_paco_dict.py --save-diag \
	-w ${w} -s ${s} \
	--tau 0.1 --mu 0.95 --kappa ${ka} --maxiter 500 \
	--diff-op data/dict-w10.txt \
	--meas-op ${outdir}/P_w${w}_random_rate_${r}_seed_${S}.txt \
	--samples  ${outdir}/${I}_samples_s${s}_w${w}_random_rate_${r}_seed_${S}.txt \
	--save-iter\
	--reference ${img} \
	--outdir ${outdir} $*
ffmpeg -i ${outdir}/iter%03d0.png -framerate 5 -vcodec copy ${outdir}/${I}_w${w}_s${s}_random_rate_${r}_seed_${S}.mkv
rm ${outdir}/iter*png
