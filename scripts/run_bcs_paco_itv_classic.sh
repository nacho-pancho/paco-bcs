#!/bin/bash
for r in {1..5}
do
  for w in 32 
  do
    for overlap in 1
    do
      let s=w-overlap
      let m=s*s
      let R=r*10
      echo ratio $R width $w stride $s dim $m meas $k
      # very strange images
      for I in lena barbara peppers mandrill goldhill
      do
        # very random random seeds 
        for S in 1863699 # 9186369 9918636 6991863 3699186 6369918 8636991
        do
          outdir="results/${I}_w${w}_s${s}_random_rate_${r}_seed_${S}"
          if [[ -f "${outdir}/recovered.png" ]]
          then
	    continue
	  fi
            ./code-bcs/bcs_measure.py -o ${outdir} -w ${w} -r ${R} -s ${s} -S ${S} -i data/image/gray/classic/${I}.png
            ./code-bcs/bcs_paco_itv.py --save-iter \
                  -w ${w} -s ${s} \
            --tau 10 --mu 0.95 --maxiter 300  \
                  --diff-op ${outdir}/D_w${w}.txt \
                  --meas-op ${outdir}/P_w${w}_random_rate_${r}_seed_${S}.txt \
                  --samples  ${outdir}/${I}_samples_s${s}_w${w}_random_rate_${rate}_seed_${S}.txt \
                  --reference data/image/classic/${I}.png \
                  --outdir ${outdir} $* | tee ${outdir}/run.log
	done # random seed
      done # image
    done # stride
  done # width
done   # ratio


