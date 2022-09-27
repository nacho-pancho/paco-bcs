#!/bin/bash
for R in {1..5}
do
  for w in 32 # 8 16 32 # 40 48
  do
    for overlap in 32 # 2 4 8 16 32
    do 
      if ((overlap > w))
      then
        continue
      fi
      let s=w-w/overlap
      let m=s*s
      let R=R*10
      echo rate $R width $w stride $s dim $m meas
      # very strange images
      for I in lena barbara peppers mandrill goldhill
      do
        # very random random seeds 
        #for r in 1863699 9186369 9918636 6991863 3699186 6369918 8636991
        for S in 9186369 9918636 6991863 3699186 6369918 8636991
        do
          outdir="results/${I}_w${w}_s${s}_random_rate_${R}_seed_${S}_paco_itv"
          if [[ -f "${outdir}/recovered.png" ]]
          then
	    continue
	  fi
	    mkdir -p ${outdir}
	    echo -e "\n\nMEASURING\n"
            ./code/bcs_measure.py -o ${outdir} --stride ${s} -w ${w} --rate ${R} -S ${S}  -i data/image/gray/classic/${I}.png
	    echo -e "\n\nRECOVERING\n"
            ./code/bcs_paco_itv.py --save-iter \
                  -w ${w} -s ${s} \
                  --tau 10 --mu 0.95 --maxiter 200 --maxiter 500  \
                  --diff-op ${outdir}/D_w${w}.txt \
                  --meas-op ${outdir}/P_w${w}_random_rate_${R}_seed_${S}.txt \
                  --samples  ${outdir}/${I}_samples_s${s}_w${w}_random_rate_${R}_seed_${S}.txt \
                  --reference data/image/gray/classic/${I}.png \
                  --outdir ${outdir} | tee ${outdir}/run.log
	done # random seed
      done # image
    done # stride
  done # width
done   # ratio


