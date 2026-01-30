# CalPerUn – Model Training

This folder contains code and instructions for training the FIERY model on pedestrian data.

---

## Overview

- Based on the official FIERY repository: https://github.com/wayveai/fiery
- Extended to support pedestrian-focused training
- Uses the NuScenes dataset following the setup described in the FIERY paper

---

## Configuration Files

New configuration files introduced for pedestrian training:

- `single_timeframe_pedestrians.yml`
- `baseline_pret_pedestrian.yml`

---

## Training Procedure

Training is performed in two stages to avoid instability.  
Training the full model from scratch with future frames may lead to NaNs.

---

### Step 1: Single Timeframe Training

Train without future frames for stabilization (8 epochs).

Command:

    python train.py --config fiery/configs/single_timeframe_pedestrians.yml

---

### Step 2: Full Model Training (with Future Frames)

Train using future frames (1 epoch).

Command:

    python train.py --config fiery/configs/baseline_pret_pedestrian.yml BATCHSIZE 3 GPUS [0,1,2,3]

Update the configuration file to load the checkpoint from Step 1.

---

## Pretrained Checkpoint

The following checkpoint is used for all experiments:

    epoch=0-step=1993_pedestrains_full.ckpt

---

## Evaluation

Generate segmentation maps for the validation dataset (used as input for experiments).

Command:

    python -u fiery/evaluate_save_res.py \
        --checkpoint ../checkpoints/epoch=0-step=1993_pedestrains_full.ckpt \
        --version trainval \
        --tag _epoch0_pedestrians \
        --save_output

Make sure the NuScenes dataset paths are correctly configured in the code.

---

## Contact

**markus.kangsepp@ut.ee**
