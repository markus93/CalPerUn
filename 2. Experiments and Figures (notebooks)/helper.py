# Import os, np, torch, torchvision, mpl, plt, F, multivariate_normal, GaussianMixture


from betacal import BetaCalibration
from sklearn.model_selection import train_test_split
import torch
import torchvision
import matplotlib as mpl
from matplotlib import pyplot as plt
from os.path import join
import os

from scipy.stats import multivariate_normal, norm
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from binnings import *


PATH_image = "images_val" # Folder name

FILE_label = "labels_val_%i.p"   # Single file name
FILE_res = "results_val_img%i.p"  # Single file name
FILE_image = "image_val_b%i_i%i_s%i_c%i.jpg"  # Single file name

def get_nr_files_in_dir(path):
    return len(os.listdir(path))

def get_file_contents(i, path_labels, path_res):
    file_label = FILE_label % i
    file_res = FILE_res % i
    
    labels = np.load(join(path_labels, file_label), allow_pickle=True)
    results = np.load(join(path_res, file_res), allow_pickle=True)
    
    return (labels, results)

def get_img(b, i, s, c):
    file_img = FILE_image  % (b, i, s, c)
        
    return Image.open(join(PATH_image, file_img))


def softmax_axis(x, axis=2):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Subtract max so biggest is 0 to avoid numerical instability
    
    # Axis 0 if only one dimensional array
    #axis = 0 if len(e_x.shape) == 1 else 1
    
    return e_x / e_x.sum(axis=axis, keepdims=True)

def get_masks(h = 60, w = 16, buffer = 2, H = 200, W = 200):  # Expects already rotated image

    mask = np.zeros(shape=(H, W))
    mask[(H//2-h):H//2, (W//2-w//2):(W//2+w//2)] = 1

    mask2 = np.zeros(shape=(200, 200))
    mask2[(H//2-h):H//2, (W//2-w//2-buffer):(W//2-w//2)] = 1  # Left side
    mask2[(H//2-h):H//2, (W//2+w//2):(W//2+w//2+buffer):] = 1  # Right side
    mask2[(H//2-h-buffer):(H//2-h), (W//2-w//2-buffer):(W//2+w//2+buffer)] = 1  # Top side
    mask2[(H//2):(H//2+buffer), (W//2-w//2-buffer):(W//2+w//2+buffer)] = 1  # Top side

    return mask.astype("bool"), mask2.astype("bool")
    
    
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

    
#denormalise_img = torchvision.transforms.Compose(
#        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         torchvision.transforms.ToPILImage(),)
#)


def plot_img(bat, ix, seq = 2):

    # Plot present RGB frames and predictions
    val_w = 2.99
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    image_ratio = (224, 480)[0] / (224, 480)[1]
    val_h = val_w * image_ratio
    fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
    width_ratios = (val_w, val_w, val_w, val_w)
    gs = mpl.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    for i in range(6):

        ax = plt.subplot(gs[i // 3, i % 3])
        showimg = get_img(bat,ix,seq,i) 

        if i > 2:
            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

        plt.imshow(showimg)

        plt.annotate(cameras[i].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
                         xycoords='axes fraction', fontsize=14)

        plt.axis('off')

    plt.show()
    
def find_instance_centers(center_prediction: torch.Tensor, conf_threshold: float = 0.1, nms_kernel_size: float = 3):
    assert len(center_prediction.shape) == 3
    center_prediction = F.threshold(center_prediction, threshold=conf_threshold, value=-1)

    nms_padding = (nms_kernel_size - 1) // 2
    maxpooled_center_prediction = F.max_pool2d(
        center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
    )

    # Filter all elements that are not the maximum (i.e. the center of the heatmap instance)
    center_prediction[center_prediction != maxpooled_center_prediction] = -1
    return torch.nonzero(center_prediction > 0)[:, 1:]
    
    
def find_gaussians(segm, center, pxl_per_obj = 5, 
                   multiplier = 100, conf_threshold = 0.1, nms_kernel_size=3, mask_threshold = 0.0001,
                  verbose = False, seed = 1001):
    
    # Prepare data
    counts = (segm*multiplier).round(0).astype(int)

    input_data = []

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            for _ in range(counts[i, j]):
                input_data.append([i, j])
    
    # Find centers
    centers = find_instance_centers(F.Tensor(center[np.newaxis, :, :]), 
                                    conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size).cpu().detach().numpy()
    
    if verbose:
        print("Centers:", centers)
    
    n_comps = len(centers)
    
    if n_comps == 0:
        print("No centers found!")
        return None, None, None, None
    
    if len(input_data) == 0:
        if verbose:
            print("No input data!")
        return None, None, None, None
    
    if len(input_data) == 1:
        if verbose:
            print("Single input data!")
        return None, None, None, None
    
    if len(input_data) < n_comps:
        if verbose:
            print("Less input data than n_comps", len(input_data), n_comps)
        return None, None, None, None
    
    GMM = GaussianMixture(n_components = n_comps, random_state = seed)
    
    #print("input:", input_data)
    
    GMM.fit(input_data)
    
    # Get distributions
    distributions = []

    for m, cov in zip(GMM.means_, GMM.covariances_):
        #print(m, cov)
        rv = multivariate_normal([m[1], m[0]], [[cov[1,1], cov[1,0]], [cov[0,1], cov[0,0]]])  # Everything in mirror
        distributions.append(rv)


    pdfs = []
    x, y = np.mgrid[0:segm.shape[1]:1, 0:segm.shape[0]:1]
    pos = np.dstack((x, y))

    mask_thresholds = []
    
    for rv in distributions:
        
        thr = rv.pdf([rv.mean[0] + 3*np.sqrt(rv.cov[0,0]), rv.mean[1] + 3*np.sqrt(rv.cov[1,1])])
        mask_thresholds.append(thr)
        pdfs.append(rv.pdf(pos))
        
    masks = []
    segm_preds = []
    #distributions

    
    ## TODO 99,9% of area not with some threshold
    

    for i, pdf in enumerate(pdfs):
        
        mask = pdf > mask_threshold #s[i]
        segm_preds.append(segm[mask.T].sum())
        
        if verbose:
            print(mask_thresholds[i])
            print("Pdf:", pdf[mask].sum())

        masks.append(mask)
        
    segm_preds = np.array(segm_preds)
        
    if verbose:
        print(segm_preds/pxl_per_obj)
    probs = np.clip(segm_preds/pxl_per_obj, 0, 1)
        
    return (probs, distributions, masks, pdfs)
    
def find_gaussians_cal(segm, segm_cal, center, pxl_per_obj = 5, 
                   multiplier = 100, conf_threshold = 0.1, nms_kernel_size=3, mask_threshold = 0.0001,
                      verbose = False, seed = 1001, clip_probs = False):
    
    # Prepare data
    counts = (segm*multiplier).round(0).astype(int)

    input_data = []

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            for _ in range(counts[i, j]):
                input_data.append([i, j])
    
    # Find centers
    centers = find_instance_centers(F.Tensor(center[np.newaxis, :, :]), 
                                    conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size).cpu().detach().numpy()
    
    if verbose:
        print("Centers:", centers)
    
    n_comps = len(centers)
    
    if n_comps == 0:
        if verbose:
            print("No centers found!")
        return None, None, None, None, None
    
    if len(input_data) == 0:
        if verbose:
            print("No input data!")
        return None, None, None, None, None
    
    if len(input_data) < n_comps:
        if verbose:
            print("Less input data than n_comps", len(input_data), n_comps)
        return None, None, None, None, None

    
    GMM = GaussianMixture(n_components = n_comps, random_state = seed)
    GMM.fit(input_data)
    
    # Get distributions
    distributions = []

    for m, cov in zip(GMM.means_, GMM.covariances_):
        #print(m, cov)
        rv = multivariate_normal([m[1], m[0]], [[cov[1,1], cov[1,0]], [cov[0,1], cov[0,0]]])  # Everything in mirror
        distributions.append(rv)


    pdfs = []
    x, y = np.mgrid[0:segm.shape[1]:1, 0:segm.shape[0]:1]
    pos = np.dstack((x, y))

    mask_thresholds = []
    
    for rv in distributions:
        
        thr = rv.pdf([rv.mean[0] + 3*np.sqrt(rv.cov[0,0]), rv.mean[1] + 3*np.sqrt(rv.cov[1,1])])
        mask_thresholds.append(thr)
        pdfs.append(rv.pdf(pos))
        
    masks = []
    segm_preds = []
    #distributions
    segm_preds_cal = []

    for i, pdf in enumerate(pdfs):
        mask = pdf > mask_thresholds[i]
        segm_preds.append(segm[mask.T].sum())
        segm_preds_cal.append(segm_cal[mask.T].sum())
        if verbose:
            print("Pdf:", pdf[mask].sum())
            print("Thresh:", mask_thresholds[i])

        masks.append(mask)
        
    segm_preds = np.array(segm_preds)
    segm_preds_cal = np.array(segm_preds_cal)
        
    if verbose:
        print("Segm:", segm_preds/pxl_per_obj)
        print("Segm cal:", segm_preds_cal/pxl_per_obj)

    probs = segm_preds/pxl_per_obj
    probs_cal = segm_preds_cal/pxl_per_obj
    
    if clip_probs:
        probs = np.clip(probs, 0, 1)
        probs_cal = np.clip(probs_cal, 0, 1)
        
    return (probs, probs_cal, distributions, masks, pdfs)
    
 
def plot_seg(idx, p, y, nr_future = 0, area = None):
    
    plt.figure(figsize=(15,7))

    plt.subplot(121)
    heatmap1 = plt.imshow(p[nr_future, idx], cmap="hot_r") #, cmap='hot', interpolation='nearest')'
    plt.colorbar(heatmap1)
    
    if area is not None:
        rect = Rectangle([area[1,0],area[0,0]],(area[1,1]-area[1,0]),(area[0,1]-area[0,0]),
                          linewidth=1,edgecolor='b',facecolor='none')
        plt.gca().add_patch(rect)

    plt.subplot(122)
    heatmap2 = plt.imshow(y[nr_future, idx], cmap="hot_r") #, cmap='hot', interpolation='nearest')'
    plt.colorbar(heatmap2)
    
    if area is not None:
        rect = Rectangle([area[1,0],area[0,0]],(area[1,1]-area[1,0]),(area[0,1]-area[0,0]),
                          linewidth=1,edgecolor='b',facecolor='none')
        plt.gca().add_patch(rect)
        

    plt.show()
    
    
def helper_fn(arr, cl_temp):
    
    idx_del = []
    for i, elem in enumerate(arr):
        for cl_elem in cl_temp:
            if np.abs(elem[0] - cl_elem[0]) <= 1 and np.abs(elem[1] - cl_elem[1]) <= 1:
                cl_temp.append(elem)
                idx_del.append(i)
                break

    arr = np.delete(arr, idx_del, 0)
            
    return arr, cl_temp

def get_clusters(arr, threshold = 0.1):

    arr = np.array(np.where(arr > threshold)).T
    clusters = []

    while len(arr) != 0:

        #print(len(arr))

        cl_temp = [arr[0]]
        arr = np.delete(arr, 0, 0)
        prev_len = len(arr)

        arr, cl_temp = helper_fn(arr, cl_temp)

        while len(arr) != prev_len:
            prev_len = len(arr)
            arr, cl_temp = helper_fn(arr, cl_temp)

        clusters.append(cl_temp)
        
    return clusters
    
    
def in_circle(xy, h1, k1, r1):
    return (xy[0] - h1)**2 + (xy[1] - k1)**2 - r1**2 < 0  #( x - h )^2 + ( y - k )^2 = r^2

def get_masks2(step = 20, len_xy = 200):
    x, y = np.mgrid[0:len_xy:1, 0:len_xy:1]
    pos = np.dstack((x, y))
    
    masks = []

    for r in range(step, 101, step):
        masks.append(np.apply_along_axis(in_circle, 1, pos.reshape(len_xy**2, 2), **{"r1":r, "h1":len_xy//2, "k1":len_xy//2}).reshape(len_xy, len_xy))

    i_max = len(masks)-1

    masks2 = masks.copy()
    masks2.append(~masks2[-1])

    #break
    # Create rings from full circles.
    for i in range(0, i_max):
        #print(i)
        if i == 0:
            temp_mask = np.logical_xor(masks[i_max-i], masks[i_max-i-1])
            masks[i_max-i] = np.logical_or(temp_mask, ~masks[i_max-i])
        else:
            masks[i_max-i] = np.logical_xor(masks[i_max-i], masks[i_max-i-1])
            
    return masks, masks2
    

def get_models_in_rings_old(p, y, Cal_Fn = BetaCalibration, nr_future = 0, test_size = 0.8, random_state = 828):

    # 20% for validation (calibration modelling)
    segm_ped_p_val, segm_ped_p_test, segm_ped_y_val, segm_ped_y_test = train_test_split(p[nr_future], 
                                                                                        y[nr_future], 
                                                                                        test_size=test_size, random_state=random_state)

    models = []
    
    masks, _ = get_masks2(step = 20)

    for i, m in enumerate(masks):

        print(i)

        ys_pxl = segm_ped_y_val[:, m].reshape(-1) 
        preds_pxl = segm_ped_p_val[:, m].reshape(-1)

        print(preds_pxl.shape)
        
        model = Cal_Fn()
        model.fit(preds_pxl, ys_pxl)

        models.append(model)
        
    return models, masks, (segm_ped_p_val, segm_ped_p_test, segm_ped_y_val, segm_ped_y_test)


def get_models_in_rings(p, y, Cal_Fn = BetaCalibration, nr_future = 0, test_size = 0.8, random_state = 828):

    # 20% for validation (calibration modelling)
    indices_val, indices_test = train_test_split(np.arange(0, p.shape[1]), test_size=test_size, random_state=random_state)

    models = []
    
    masks, _ = get_masks2(step = 20)
    
    segm_ped_p_val = p[nr_future, indices_val,]
    segm_ped_y_val = y[nr_future, indices_val,]

    for i, m in enumerate(masks):

        print(i)

        ys_pxl = segm_ped_y_val[:, m].reshape(-1) 
        preds_pxl = segm_ped_p_val[:, m].reshape(-1)

        print(preds_pxl.shape)
        
        model = Cal_Fn()
        model.fit(preds_pxl, ys_pxl)

        models.append(model)
        
    return models, masks, (indices_val, indices_test)




def generate_rel_diagrams(preds, ys, equal_size = False, n_bins=10, title_str = ""):
    
    if equal_size:
        binning_method = EqualSizeBinning
    else:
        binning_method = EqualWidthBinning        
        
    
       
    preds_test_cal = preds
    ys_test = ys

    fig, ax = plt.subplots(figsize=(3.7,3.7))
    binning = binning_method(preds_test_cal, ys_test, None, n_bins=n_bins)
    print("\nECE_abs:", binning.ECE_abs)
    polygons = binning.construct_plt_polygons_slope_1()

    if not equal_size:
        plt.xlim(0,1)
        plt.ylim(0,1)
    else:
        plt.yscale('log')
        plt.xscale('log')

    plt.xlabel("Predicted probability", fontsize=12)
    plt.ylabel("Calibrated probability", fontsize=12)
    plt.title("nr future = 0%s" % title_str) #plt.title("nr_future = %i" % f)

    for polygon in polygons:
        polygon.set_alpha(1)
        ax.add_line(polygon)

    plt.plot([0,1], [0,1], "--", c="black", label="Main diagonal", zorder=2)
    print([len(binning.binned_ids[i]) for i in range(n_bins)])
    print("Conf:", [np.mean(binning.binned_p[i]) for i in range(n_bins)])
    print("Acc:", [np.mean(binning.binned_y[i]) for i in range(n_bins)])
    
    
def get_results(path_labels, path_res, results_name = "segmentation", label_name = "segmentation", 
                 flip_input=True, use_softmax = True):
    
    nr_files = get_nr_files_in_dir(path_labels)
    print("Files:", nr_files)
    
    #assert get_nr_files_in_dir(path_labels) == get_nr_files_in_dir(PATH_res_pedestrians) 
    
    labels, results = get_file_contents(0, path_labels, path_res)
    
    if use_softmax:
        segm_p = softmax_axis(results[0][results_name], axis=2)
    else:
        segm_p = results[0][results_name]
    segm_p = segm_p[:, :, -1,]
    
    segm_ps = [[] for i in range(segm_p.shape[1])]
    segm_ys = [[] for i in range(segm_p.shape[1])]
    segm_logs = [[] for i in range(segm_p.shape[1])]
    inst_ys = [[] for i in range(segm_p.shape[1])]


    
    for n_file in range(nr_files):        
        
        if(n_file%100 == 0):
            print(n_file)
        
        labels, results = get_file_contents(n_file, path_labels, path_res)
    
        segm_y = labels.item()[label_name]
        segm_y = segm_y[:,:,-1,]
        
        inst_y = labels.item()["instance"]        
        
        if use_softmax:
            segm_p = softmax_axis(results[0][results_name], axis=2)
        else:
            segm_p = results[0][results_name]
        segm_p = segm_p[:, :, -1,]
        
        segm_log = results[0][results_name]  # # nr of images, nr futures, class, H, W 
        
        if flip_input:
            segm_y = np.flip(segm_y, axis=(2,3))
            segm_p = np.flip(segm_p, axis=(2,3))
            inst_y = np.flip(inst_y, axis=(2,3))
            segm_log = np.flip(segm_log, axis=(3,4))  # # nr of images, nr futures, class, H, W 
            
        for b in range(segm_p.shape[0]):
            for n_future in range(segm_p.shape[1]):
                segm_ps[n_future].append(segm_p[b, n_future])
                segm_ys[n_future].append(segm_y[b, n_future])
                segm_logs[n_future].append(segm_log[b, n_future])
                inst_ys[n_future].append(inst_y[b, n_future])
                
    segm_ps = np.array(segm_ps)
    segm_ys = np.array(segm_ys)
    segm_logs = np.array(segm_logs)
    inst_ys = np.array(inst_ys)
                        
    return segm_ys, segm_ps, segm_logs, inst_ys # ys, preds, 
    
    
def rot(theta):
    theta = np.deg2rad(theta)
    
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])