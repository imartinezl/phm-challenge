#!/bin/bash
#SBATCH -J main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=abacus002
#SBATCH --cpus-per-task=8
#SBATCH --array=1-96

# salloc --ntasks=1 --partition=interactive --time=1-00:00:00  --mem=12GB --gpus=1 --nodelist=abacus002 --cpus-per-task=8

module load Miniconda3/4.9.2
# if ! (conda env list | grep ".venv2") ; then 
# 	conda create --name .venv2 python=3.7 -y
# fi
module load CUDA/11.1.1-GCC-10.2.0
source activate .venv2
export CXX=g++

# pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

CASE_NUM=`printf %03d $SLURM_ARRAY_TASK_ID`

cd runs
srun bash run$CASE_NUM.sh