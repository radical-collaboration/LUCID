#!/bin/bash
#SBATCH --job-name=cellSAM      ## Name of the job
#SBATCH --output=test_all_3_17_week2_new.out    ## Output file
#SBATCH --partition csi
#SBATCH --time=24:00:00         ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8     ## The number of threads the code will use
#SBATCH --gres=gpu:1            ## Real memory(MB) per CPU required by the job.
#SBATCH -A csiml  


export https_proxy=http://proxy.sdcc.bnl.local:3128/
python test_all_img_all_week.py --data_dir '/hpcgpfs01/scratch/xyu1/cell_data/cellpaint/rpe_images/week_two/'