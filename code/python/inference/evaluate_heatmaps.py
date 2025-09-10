import numpy as np
import pandas as pd
import pickle
from KDEpy import FFTKDE
from sklearn.neighbors import KernelDensity
from skimage.measure import block_reduce
import cv2
import os
import time


def load_data(condition=None, split="test", rt_clean=True):

    assert condition in {"vision", "audio", "occluded"}

    if rt_clean:
        add_str = "_rt_cleaned"
    else:
        add_str = ""

    eye_data = pd.read_pickle(f"../../../data/human_data/inference/df_{condition}{add_str}.xz")

    if split == "train":
        eye_data = eye_data[eye_data['participant'] < 16]

    # Clean dtat to exclude blinks or eye-movment outside of the plinko box
    eye_data = eye_data[(eye_data['x'] > 0) & (eye_data['x'] < 600) & (eye_data['y'] > 0) & (eye_data['y'] < 500)]

    return eye_data


def load_model_perf(model_version):

    with open(model_version, "rb") as f:
        model_perf = pickle.load(f)


    return model_perf


def load_human_heatmap(condition, trial, split="test", cut=None):

    if cut is None:
        str_add = ""
    else:
        str_add = f"_cut_{cut}"

    str_add += "_rt_cleaned"

    path = f"heatmaps/human_trial_{condition}_{split}_{trial}{str_add}.pickle"

    with open(path, "rb") as f:
        human_hm = pickle.load(f)
        
    return human_hm


def make_kde_grid(grid_step):

    col_num = int(np.ceil(601/grid_step))
    row_num = int(np.ceil(501/grid_step))

    kde_grid = np.zeros((row_num*col_num, 2))
    for x in range(col_num):
        for y in range(row_num):
            grid_row = x*row_num+y
            kde_grid[grid_row,0] = x*grid_step
            kde_grid[grid_row,1] = y*grid_step


    return kde_grid, row_num, col_num


def make_kde(trial_looks, grid_step, bw=50, method="FFT"):

    kde_grid, row_num, col_num = make_kde_grid(grid_step)

    if method == "FFT":

        kde = FFTKDE(kernel="gaussian", bw=bw).fit(trial_looks)
        histogram = kde.evaluate(kde_grid)

    elif method == "scikit":

        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(trial_looks)
        histogram = np.exp(kde.score_samples(kde_grid))

    else:
        raise Exception('Method "{}" not implemented.'.format(method))


    histogram = np.flip(histogram.reshape(col_num,row_num).T, axis=0)

    histogram /= np.sum(histogram)

    return histogram

def gaussian_kernel(center, size=(501, 601), sigma=50):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)[:, np.newaxis]
    x0, y0 = center
    kernel = np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / sigma**2)
    return kernel

def create_count_histogram(points, size=(501, 601), sigma=100):
    hist = np.ones(size)
    for point in points:
        hist += gaussian_kernel(point, size=size, sigma=sigma)

    hist = np.flipud(hist)
    return hist

def convert_arr(arr, grid_step):
    rows, cols = arr.shape
    
    sig = []
    for r in range(rows):
        for c in range(cols):
            v = float(arr[r,c])
            sig.append([v,r*grid_step,c*grid_step])
            
    return np.array(sig, dtype=np.float32)

def reduce_hist(hist, grid_step=20):
    coarse_hist = block_reduce(hist, (grid_step, grid_step), np.mean)
    coarse_hist /= np.sum(coarse_hist)
    return coarse_hist

def compute_emd(model_hist, human_hist, grid_step=20):
    
    coarse_model_hm = reduce_hist(model_hist, grid_step)
    model_sig = convert_arr(coarse_model_hm, grid_step)

    coarse_human_hm = reduce_hist(human_hist, grid_step)  
    human_sig = convert_arr(coarse_human_hm, grid_step)
    
    dist, _, _ = cv2.EMD(model_sig, human_sig, cv2.DIST_L2)
    
    return dist

world_files = os.listdir("stimuli/ground_truth/")
world_nums = sorted([int(file[6:-5]) for file in world_files])

def compute_emd_all_trials(model_pred, condition, labels=None, world_nums=world_nums, cut=None):
    tr_len = 501*601
    
    model_emd = []
    
    for i, tr_num in enumerate(world_nums):
        
        print("Trial:", tr_num)
        
        tr_start = i*tr_len
        tr_pred = model_pred[tr_start:tr_start + tr_len]
        if labels is not None:
            tr_labels = labels[tr_start:tr_start + tr_len]
        
        tr_pred[tr_pred < 0] = 0
        tr_pred /= np.sum(tr_pred)
        
        model_hm = tr_pred.reshape(501, 601)[:500, :600]

        if labels is not None:
            human_hm = tr_labels.reshape(501, 601)[:500, :600]
        else:
            human_hm = load_human_heatmap(condition, tr_num, split="test", cut=cut)[:500, :600]
        
        dist = compute_emd(model_hm, human_hm)
        
        model_emd.append(dist)
    
    return model_emd


def compute_emd_human_data(condition1, condition2, world_nums=world_nums, cut=None):

    human_emd = []

    for tr_num in world_nums:

        print("Trial:", tr_num)

        human_hm1 = load_human_heatmap(condition1, tr_num, split="test", cut=cut)[:500, :600]
        coarse_human_hm1 = block_reduce(human_hm1, (20,20), np.mean)
        coarse_human_hm1 /= np.sum(coarse_human_hm1)

        human_hm2 = load_human_heatmap(condition2, tr_num, split="test", cut=cut)[:500, :600]
        coarse_human_hm2 = block_reduce(human_hm2, (20,20), np.mean)
        coarse_human_hm2 /= np.sum(coarse_human_hm2)

        dist = compute_emd(coarse_human_hm1, coarse_human_hm2)

        human_emd.append(dist)

    return human_emd