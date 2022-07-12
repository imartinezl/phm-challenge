#!/bin/bash
#SBATCH -J main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=abacus001
#SBATCH --cpus-per-task=8

# salloc --ntasks=1 --partition=interactive --time=60  --mem=4GB --gpus=1 --nodelist=abacus001 --cpus-per-task=8

module load Miniconda3/4.9.2
# if ! (conda env list | grep ".venv2") ; then 
# 	conda create --name .venv2 python=3.7 -y
# fi
module load CUDA/10.2.89-GCC-8.3.0 # for CPAB
source activate .venv

srun python run.py
