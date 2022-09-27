#!/bin/bash
for R in {1..5}
do
  for w in 8 16 32 # 40 48
  do
    for overlap in 2 4 8 16 32
    do
      if ((overlap > w))
      then
	continue
      fi
      let s=w-w/overlap
      let m=s*s
      let k=m*R/10
      if ((k < 2))
      then
        continue
      fi
      # very strange images
      for K in {01..23}
      do
	I=kodim${K}
        # very random random seeds 
        for r in 18636998 # 9186369 9918636 6991863 3699186 6369918 8636991
        do
          outdir="results/kodak/${I}_w${w}_s${s}_random_k${k}_r${r}"
          outfile="${outdir}/recovered.png"
          optfile="${outdir}/optimization.txt"
	  ref="data/image/gray/kodak/${I}.png" 
	  echo -n -e "\t${R}"
	  echo -n -e "\t${w}"
	  echo -n -e "\t${s}"
    	  echo -e -n "\t${m}\t${s}\t${k}"
	  echo -n -e "\t${K}"
	  echo -n -e "\t${r}"
	  if [[ -f $optfile ]]
          then
		  echo -n -e "\t"
		  ./code-bcs/ssim.py $ref $outfile
	  else
		  echo -e "\tnan"
	  fi
	done # random seed
      done # image
    done # stride
  done # width
done   # ratio


