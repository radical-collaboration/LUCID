#!/bin/bash
#SBATCH --job-name=cellSAM      ## Name of the job
#SBATCH --output=cellSAM.out    ## Output file
#SBATCH --partition csi
#SBATCH -q csi
#SBATCH --time=24:00:00         ## Job Duration
#SBATCH --ntasks=6              ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8       ## The number of threads the code will use
#SBATCH --gres=gpu:4
#SBATCH -A csiml  


export https_proxy=http://proxy.sdcc.bnl.local:3128/
# NOTE: activate virtual/conda environment first

CP_RUN_TIME=$(squeue -j "$SLURM_JOB_ID" -h -O TimeLimit | awk -F: '{ print ($1 * 60) + $2 }')
python3 cell.rp.py -t $CP_RUN_TIME -c ic2.json
# python test_all.py \
#  --data_dir '/hpcgpfs01/scratch/xyu1/cell_data/cellpaint/rpe_images/week_two/'
