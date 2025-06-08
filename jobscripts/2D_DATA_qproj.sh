#!/bin/bash
#BSUB -J 2D_qproj
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -o /zhome/4c/1/203176/BayesWithoutUnderfitting/hpc_logs/OUTPUT_FILE_2D_qproj.out
#BSUB -e /zhome/4c/1/203176/BayesWithoutUnderfitting/hpc_logs/OUTPUT_FILE_2D_qproj.err

module load python3/3.11.9

source /zhome/4c/1/203176/BayesWithoutUnderfitting/venv/bin/activate
python /zhome/4c/1/203176/BayesWithoutUnderfitting/main.py --experiment 2d_altproj_qproj > /zhome/4c/1/203176/BayesWithoutUnderfitting/hpc_logs/OUTPUT_FILE_2D_qproj.log 2>&1