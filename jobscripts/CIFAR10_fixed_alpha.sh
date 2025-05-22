#!/bin/bash
#BSUB -J CIFAR10_ALPHA_10
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -o /zhome/4c/1/203176/BayesWithoutUnderfitting/hpc_logs/OUTPUT_FILE_CIFAR10_ALPHA_10.out
#BSUB -e /zhome/4c/1/203176/BayesWithoutUnderfitting/hpc_logs/OUTPUT_FILE_CIFAR10_ALPHA_10.err

module load python3/3.11.9

source /zhome/4c/1/203176/BayesWithoutUnderfitting/venv/bin/activate
python /zhome/4c/1/203176/BayesWithoutUnderfitting/main.py --experiment cifar10 > /zhome/4c/1/203176/BayesWithoutUnderfitting/hpc_logs/OUTPUT_FILE_CIFAR10_ALPHA_10.log 2>&1