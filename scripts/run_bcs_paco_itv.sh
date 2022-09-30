#!/bin/bash
for r in {1..5}
do
  for w in 32 # 8 16 32 # 40 48
  do
    for overlap in 1 
    do 
      let s=w-overlap
      let m=s*s
      let R=r*10
      echo rate $R width $w stride $s dim $m meas
      # very strange images
      for K in 01 07 09 15 16 19 23
      do
	I=kodim${K}
        # very random random seeds 
        #for r in 1863699 9186369 9918636 6991863 3699186 6369918 8636991
        for S in 9186369 # 9918636 6991863 3699186 6369918 8636991
        do
	  echo "IMAGE ${I} RATE ${R} WIDTH ${w} STRIDE ${s} SEED ${S}"
          outdir="results/${I}_w${w}_s${s}_random_rate_${R}_seed_${S}_paco_itv"
          if [[ -f "${outdir}/recovered.png" ]]
          then
	    echo "${outdir}: Already computed"
	    continue
	  fi
	    mkdir -p ${outdir}
	    echo -e "\n\nMEASURING\n"
            ./code/bcs_measure.py -o ${outdir} --stride ${s} -w ${w} --rate ${R} -S ${S}  -i data/image/gray/kodak/${I}.png
	    echo -e "\n\nRECOVERING\n"
            ./code/bcs_paco_itv.py --save-iter \
                  -w ${w} -s ${s} \
            --tau 10 --mu 0.95 --maxiter 200 --outer-tol 1e-3  \
                  --diff-op ${outdir}/D_w${w}.txt \
                  --meas-op ${outdir}/P_w${w}_random_rate_${R}_seed_${S}.txt \
                  --samples  ${outdir}/${I}_samples_s${s}_w${w}_random_rate_${R}_seed_${S}.txt \
                  --reference data/image/gray/kodak/${I}.png \
                  --outdir ${outdir} | tee ${outdir}/run.log
	done # random seed
      done # image
    done # stride
  done # width
done   # ratio


