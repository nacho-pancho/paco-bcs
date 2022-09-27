#!/bin/bash
for R in {1..5}
do
  for w in 32 # 40 48
  do
    for overlap in 32 # 4 8 16
    do
      if ((overlap > w))
      then
	continue
      fi
      let s=w-w/overlap
      let m=s*s
      let k=m*R/10
      let rate=R*10
      # very strange images
      for I in barbara barbara peppers mandrill goldhill
      do
        # very random random seeds 
        for r in 9186369 9918636 6991863 3699186 6369918 8636991
        do
          outdir="results/${I}_w${w}_s${s}_random_rate_${rate}_seed_${r}_paco_itv"
          outfile="${outdir}/recovered.png"
          optfile="${outdir}/optimization.txt"
	  ref="data/image/gray/classic/${I}.png" 
	  echo -n -e "\t${rate}"
	  echo -n -e "\t${w}"
	  echo -n -e "\t${s}"
    	  echo -e -n "\t${m}"
	  echo -n -e "\t${I}"
	  echo -n -e "\t${r}"
	  if [[ -f $optfile ]]
          then
		  echo -n -e "\t"
		  ./code/ssim.py $ref $outfile
	  else
		  echo -e "\tnan"
	  fi
	done # random seed
      done # image
    done # stride
  done # width
done   # ratio


