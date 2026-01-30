#!/usr/bin/bash


# The module with all the NMT / deep learning packages

# export PATH=/gpfs/hpc/home/markus93/miniconda3_gpu/bin:$PATH

#SBATCH -J Fiery_training

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#The maximum walltime of the job is a 48 hours
#SBATCH -t 24:00:00

#SBATCH --mem=180G

#Leave this here if you need a GPU for your job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4


export PATH=/home/hpc_markus93/miniconda3/bin:$PATH
source activate fiery

echo "Fiery training on trainval batch_size 12, GPUs 4 (3 per gpu)"
python train.py --config fiery/configs/baseline_pret_pedestrian.yml BATCHSIZE 3 GPUS [0,1,2,3]
echo "Job finished"
