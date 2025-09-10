import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

import visual
import utils

rgb_green = (17/255, 119/255, 51/255)
rgb_skyblue = (136/255, 204/255, 238/255)
rgb_magenta = (170/255, 68/255, 153/255)

def softmax(arr, axis=1):
    expo = np.exp(arr)
    expo_sum = np.sum(expo, axis=axis)
    expo_norm = expo/expo_sum[:,np.newaxis]
    
    return expo_norm

def make_sig(arr):
    dist_len = arr.shape[0]
    
    sig = []
    for x in range(dist_len):
        v = float(arr[x])
        sig.append([v,x])
            
    return np.array(sig, dtype=np.float32)

def compute_kdes(df, x_label):
    
    kde_grid = np.arange(600)
    
    world_set = df["world"].unique()
    holes = [1,2,3]
    
    kde_dict = {}
    
    for world in world_set:
        world_dict = {}
        
        for hole in holes:
            
            samples = df[(df["world"] == world) &
                         (df["hole"] == hole)][x_label].to_numpy()
            
            kde = FFTKDE(kernel="gaussian", bw=20).fit(samples)
            hist = kde.evaluate(kde_grid)
            
            world_dict[hole] = hist
            
        kde_dict[world] = world_dict
        
    return kde_dict

def compute_kdes_mat(data_mat, kde_size=600):
    rows, _ = data_mat.shape
    
    kde_mat = np.zeros((rows, kde_size))
    x_grid = np.arange(kde_size)
    
    for i in range(rows):
        data = data_mat[i,:]
        kde = FFTKDE(kernel="gaussian", bw=20).fit(data).evaluate(x_grid)
        
        kde_mat[i,:] = kde
        
    return kde_mat

def make_kde_dict(pred_mat, world_list):
    kde_dict = {}

    for i, world_num in enumerate(world_list):
        world_dict = {}

        for hole in [1,2,3]:
            row = i*3 + (hole-1)
            kde = pred_mat[row,:]
            world_dict[hole] = kde

        kde_dict[world_num] = world_dict

    return kde_dict

def compute_emds(model_kdes, human_kdes, world_set, emd_method="scipy"):
    
    emd_dict = {"world": [],
                "hole": [],
                "EMD": []}

    for world in world_set:
        for hole in [1,2,3]:
            hist1 = human_kdes[world][hole]
            hist2 = model_kdes[world][hole]

            if emd_method == "cv":
                sig1 = make_sig(hist1)
                sig2 = make_sig(hist2)

                emd = EMD(sig1, sig2, DIST_L2)[0]
            
            elif emd_method == "scipy":
                emd = wasserstein_distance(np.arange(600), np.arange(600), u_weights=hist1, v_weights=hist2)

            emd_dict["world"].append(world)
            emd_dict["hole"].append(hole)
            emd_dict["EMD"].append(emd)
        
    df_emd = pd.DataFrame(emd_dict)
    
    return df_emd

def plot_kdes(kde_dict, world_num, multiplier=8000, colors=[rgb_green, rgb_skyblue, rgb_magenta], alpha=0.8, plot_gt=False, format="narrow"):

    kdes = kde_dict[world_num]

    if format == "narrow":
        # fig, ax = plt.subplots(1,1, figsize=(5,5))

        ax = visual.show_unity_trial(world_num, exp="prediction")

        for i, hole in enumerate([1,2,3]):

            kde = kdes[hole]
            ax = visual.graph_conditional_dist(ax, kde, multiplier=multiplier, col=colors[i], precomputed=True, alpha=alpha, zorder=0)

            trial = utils.load_trial(world_num, experiment="prediction", hole=hole)
            trial["drop_noise"] = 0
            trial["collision_noise_mean"] = 1.0
            trial["collision_noise_sd"] = 0.0

            if plot_gt:
                
                xs, ys, fp = visual.simulate_drop(trial, i)
                ax = visual.draw_drop(ax, i, trial, xs, ys, fp, colors=colors)
            else:
                ax = visual.draw_circle_drop(ax, i, trial, colors=colors)

    elif format == "wide":
        fig, ax = plt.subplots(1,3, figsize=(15,5))

        for i, hole in enumerate([1,2,3]):
            
            sub_ax = ax[i]
            kde = kdes[hole]
            
            sub_ax = visual.show_unity_trial(world_num, exp="prediction", hole=hole, ax=sub_ax)
            sub_ax = visual.graph_conditional_dist(sub_ax, kde, multiplier=multiplier, col=colors[i], precomputed=True, alpha=alpha)

    else:
        raise Exception(f"Format {format} not implemented.")
        
    return ax