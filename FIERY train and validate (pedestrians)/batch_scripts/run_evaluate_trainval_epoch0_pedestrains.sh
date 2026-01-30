#!/usr/bin/bash


# The module with all the NMT / deep learning packages

# export PATH=/gpfs/hpc/home/markus93/miniconda3_gpu/bin:$PATH

export PATH=/home/hpc_markus93/miniconda3/bin:$PATH
source activate fiery

echo "Fiery evaluation pedestrians val set"
python -u evaluate_save_res.py --checkpoint nuscenes/checkpoints_pedestrians/epoch=0-step=1993_pedestrains_full.ckpt --version trainval --tag _epoch0_pedestrians --save_output 
echo "Job finished"
