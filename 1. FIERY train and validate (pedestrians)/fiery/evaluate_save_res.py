from argparse import ArgumentParser

import torch
from tqdm import tqdm

from fiery.data import prepare_dataloaders
from fiery.trainer_res import TrainingModule
from fiery.metrics import IntersectionOverUnion, PanopticMetric
from fiery.utils.network import preprocess_batch
from fiery.utils.instance import predict_instance_segmentation_and_trajectories

import pickle
import numpy as np
from os.path import join

import torch
#import torch.nn as nn
import torchvision

import matplotlib as mpl
from PIL import Image
import os

import time



class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

    
denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
)

# 30mx30m, 100mx100m
EVALUATION_RANGES = {'30x30': (70, 130),
                     '100x100': (0, 200)
                     }
                     
PATH_img = "images_val%s"
PATH_res = "results_val%s"
PATH_labels = "labels_val%s"


def eval(checkpoint_path, dataroot, version, save_img = False, save_output = False, save_labels = False, tag = ""):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    #torch.cuda.set_per_process_memory_fraction(0.5, 0)
    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 4  # NB! was 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    _, valloader = prepare_dataloaders(cfg)

    panoptic_metrics = {}
    iou_metrics = {}
    n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    for key in EVALUATION_RANGES.keys():
        panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to(
            device)
        iou_metrics[key] = IntersectionOverUnion(n_classes).to(device)
        
    results_save = []    
    

    for i, batch in enumerate(tqdm(valloader)):
    
        start_time = time.time()
    
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        if save_img:
        
            print("Img shape:", image.shape, "- i:", i)
            img_A = np.array(image.cpu().detach().numpy()) #.dump()
                    
            for e in range(img_A.shape[0]):
                for s in range(img_A.shape[1]):
                    for c in range(img_A.shape[2]):                  
                
                        im = denormalise_img(torch.tensor(img_A[e, s, c,]))
                        im.save(join(PATH_img % tag, "image_val_b%i_i%i_s%i_c%i.jpg" % (i, e, s, c)))
            
            continue
        
        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        start_time2 = time.time()

        with torch.no_grad():
            # Evaluate with mean prediction
            noise = torch.zeros((batch_size, 1, model.latent_dim), device=device)
            output, output_np = model(image, intrinsics, extrinsics, future_egomotion,
                           future_distribution_inputs, noise=noise)
                           

            
        print("--- %s seconds (model run time) ---" % (time.time() - start_time2))


        # Consistent instance seg
        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            output[0], compute_matched_centers=False, make_consistent=True
        )

        segmentation_pred = output[0]['segmentation'].detach()
        segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

        for key, grid in EVALUATION_RANGES.items():
            limits = slice(grid[0], grid[1])
            panoptic_metrics[key](pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
                                  labels['instance'][..., limits, limits].contiguous()
                                  )

            iou_metrics[key](segmentation_pred[..., limits, limits].contiguous(),
                             labels['segmentation'][..., limits, limits].contiguous()
                             )
                             
                                                              
        if save_output:      
            print("seg:", np.array(output_np)[0]["segmentation"][0,0])
            np.array(output_np).dump(join(PATH_res % tag, "results_val_img%i.p" % i))
        
        if save_labels:
        
            labels_np = {"segmentation":labels["segmentation"].cpu().detach().numpy(),
                        'instance':labels["instance"].cpu().detach().numpy(), 
                        'centerness':labels["centerness"].cpu().detach().numpy(), 
                        'flow':labels["flow"].cpu().detach().numpy(),
                        'offset':labels["offset"].cpu().detach().numpy()
                        }                    
                    
            np.array(labels_np).dump(join(PATH_labels % tag,"labels_val_%i.p" % i))
            
        
        print("--- %s seconds (whole loop) ---" % (time.time() - start_time))
    

    results = {}
    for key, grid in EVALUATION_RANGES.items():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

        iou_scores = iou_metrics[key].compute()
        results['iou'] = results.get('iou', []) + [100 * iou_scores[1].item()]

    for panoptic_key in ['iou', 'pq', 'sq', 'rq']:
        print(panoptic_key)
        print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery evaluation, save images, results and labels')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='./nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval', 'test'],  # added test
                        help='dataset version')
    parser.add_argument('--tag', default='_temp', type=str, help='path identification tag, to differentiate between versions')
    parser.add_argument('--save_img', help='Save image files.', action='store_true')
    parser.add_argument('--save_output', help='Save models predictions/output.', action='store_true')
    parser.add_argument('--save_labels', help='Save true labels.', action='store_true')
    
    args = parser.parse_args()
    
    # Create folders
    
    if args.save_img:
        if not os.path.exists(PATH_img % args.tag):
            os.makedirs(PATH_img % args.tag)
        
    if args.save_output:
        if not os.path.exists(PATH_res % args.tag):
            os.makedirs(PATH_res % args.tag)    
    if args.save_labels:
        if not os.path.exists(PATH_labels % args.tag):
            os.makedirs(PATH_labels % args.tag)    


    eval(args.checkpoint, args.dataroot, args.version, args.save_img, args.save_output, args.save_labels, args.tag)
