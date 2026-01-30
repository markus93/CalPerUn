#!/usr/bin/bash


# The module with all the NMT / deep learning packages

# export PATH=/gpfs/hpc/home/markus93/miniconda3_gpu/bin:$PATH

#SBATCH -J Fiery_training

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#The maximum walltime of the job is a 8 hours
#SBATCH -t 08:00:00

#SBATCH --mem=120G

#Leave this here if you need a GPU for your job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4


export PATH=/home/hpc_markus93/miniconda3/bin:$PATH
source activate fiery

echo "Fiery training on trainval batch_size 4, GPUs 4, temporal single timeframe, train on pedestrians"
python train.py --config fiery/configs/single_timeframe_pedestrians.yml
echo "Job finished"
