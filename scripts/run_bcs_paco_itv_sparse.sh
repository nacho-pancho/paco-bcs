#!/bin/bash
for R in {1..9}
do
  for w in 12 16 24 32 # 40 48
  do
    s=8
    while ((s < $w))
    do
      let m=s*s
      let k=m*R/10
      echo ratio $R width $w stride $s dim $m meas $k
      # very strange images
      for K in 01 07 09 15 16 19 23
      do
	      I=kodim${K}
        # very random random seeds 
        for r in 1863699 9186369 9918636 6991863 3699186 6369918 8636991
        do
          outdir="results/${I}_w${w}_s${s}_random_k${k}_r${r}"
          if [[ -f "${outdir}/recovered.png" ]]
          then
	    continue
	  fi
            ./code-bcs/bcs_measure.py -o ${outdir} -w ${w} -k ${k} -s ${s} -r ${r} -i data/image/gray/kodak/${I}.png
            ./code-bcs/bcs_paco_itv.py --save-diag \
                  -w ${w} -s ${s} \
            --tau 10 --mu 0.95 --maxiter 200 --outer-tol 1e-3 --inner-tol 1e-6  \
                  --diff-op ${outdir}/D_w${w}.txt \
                  --meas-op ${outdir}/P_w${w}_random_k${k}_r${r}.txt \
                  --samples  ${outdir}/${I}_samples_s${s}_w${w}_random_k${k}_r${r}.txt \
                  --reference data/image/gray/kodak/${I}.png \
                  --outdir ${outdir} $* #| tee ${outdir}/run.log
	done # random seed
      done # image
      let s=s+8
    done # stride
  done # width
done   # ratio


