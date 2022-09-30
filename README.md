# PACO-BCS

This repository was created as supporting material for the following conference paper:

I.Ramírez, "Block Compressive Sensing using Patch Consensus"

The contents are organized as follows:

* **code**: source code (Python)
* **data**: input data. The repository contains two shell scripts for downloading the data used in the experiments of the paper.
* **scripts**: there are four types of scripts (`demo`,`run`,`summarize` and `show`), for three different variants of the PACO-BCS (`itv`,`dct`,`dict`). See **Running the code** below for a description.
* **results**: the output of the algorithm goes here

## Running the code

### Variants

   * **itv**: Total Variation BCS; this is the version described in the paper
   * **dct**: DCT-based implementation, not described in the paper; works quite well
   * **dict**: BCS based on a learned sparse model (dictionary); this does not work well for the time being

### Scripts

   * **demo_bcs_paco_<variant>.sh**: run PACO-BCS on a single image (Lena by default). This  script runs `bcs_measure.py`, which creates a compressed representation of the image, and then `bcs_paco_<variant>.py` to recover it using the specified variant.
   * **run_bcs_paco_<variant>.sh**: these scripts run the full suite of experiments, for the various parameters of the PACO-BCS algorithm, over all the images. There is one for the Kodak dataset, and another for the classic dataset (Lena, Barbara, etc.)
   * **summarize_bcs_paco_<variant>.sh**: scans the results obtained using the `run` scripts and produces a table with MSE, PSNR and SSIM results. This table can be loaded as a CSV.
   * **show_bcs_paco_<variant>.py*: these take the output produce by the `summarize` scripts and produce the figures and tables shown in the paper

### Source code

   * `bcs_measure.py`: takes an image and a set of sensing parameters (block size, random seed, number of projections, etc.) and produces a CS representation of each block in the image. The output of this script is the input to the following ones.
   * `bcs_paco_<variant>.py`: main PACO-BCS implementation for each variant
   * `bcs_spl.py`: implements the Smooth Projected Landweber heuristic; used for comparison
   * `operators.py`: utility to generate difference operators such as TV in matrix form
   * `patch_mapping.py`: extraction and stitching operators. *This is actually the main bottleneck of the algorithm*. Although we have a much faster implementation using Python's C interface, we have not included it in this repository as it complicates portability and reproducibility significantly. 
   
