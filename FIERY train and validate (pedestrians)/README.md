# CalPerUn
Official repository of the article "Calibrating Perception Uncertainty for Autonomous Driving"


## Model training:

-> Official FIERY repository (https://github.com/wayveai/fiery?tab=readme-ov-file) used as base, added the feature to train on pedestrians
-> NuScene data used for training as in FIERY article


-> new config files for training pedestrians: single_timeframe_pedestrians.yml & baseline_pret_pedestrian.yml 


-> Training done in two steps to avoid instability in training (training full model from scratch introduces "nan"-s):
	1) Without future frames (8 epochs)
		-> python train.py --config fiery/configs/single_timeframe_pedestrians.yml
	2) Also with future frames (1 epoch)
		-> python train.py --config fiery/configs/baseline_pret_pedestrian.yml BATCHSIZE 3 GPUS [0,1,2,3]
		-> change the config yml to use previously trained checkpoint.
		
	
-> Pretrained checkpoint (epoch=0-step=1993_pedestrains_full.ckpt) used for the experiments


-> Evaluate model to get segmentation maps for validation data (data used as an input for the experiments)
	-> python -u fiery/evaluate_save_res.py --checkpoint ../checkpoints/epoch=0-step=1993_pedestrains_full.ckpt --version trainval --tag _epoch0_pedestrians --save_output 
	-> check that PATH's for NuScene dataset matches with code

## Contacts: markus.kangsepp@ut.ee