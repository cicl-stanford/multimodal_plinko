import numpy as np
import pandas as pd
import evaluate_heatmaps as e
from sklearn.linear_model import LinearRegression
import pickle
import time
import os

def load_model_perf(model_version):
    with open(model_version, "rb") as f:
        model_perf = pickle.load(f)
        
    return model_perf

def make_events_heatmaps(trial_events, model_type, hist_type="kde"):
    if model_type == "bandit" or model_type == "mixed":
        drop_pts, col_obs_pts, col_wall_pts, col_ground_pts = [], [], [], []
        for run in trial_events:
            drop_pts += run['drop']
            col_obs_pts += run['col_obs']
            col_wall_pts += run['col_wall']
            col_ground_pts += run['col_ground']
    elif model_type == "uniform" or model_type == "smart" or model_type == "sequential":
        # print(trial_events)
        # print(len(trial_events))
        # print()
        drop_pts, col_obs_pts, col_wall_pts, col_ground_pts = trial_events['drop'], trial_events['col_obs'], trial_events['col_wall'], trial_events['col_ground']
    else:
        raise Exception("Model {} not implemented.".format(model_type))
            
    if hist_type=="kde":
        drop_hist = e.make_kde(np.array(drop_pts), 1) if len(drop_pts) != 0 else np.zeros((501, 601))
        obs_hist = e.make_kde(np.array(col_obs_pts), 1) if len(col_obs_pts) != 0 else np.zeros((501, 601))
        wall_hist = e.make_kde(np.array(col_wall_pts), 1) if len(col_wall_pts) != 0 else np.zeros((501, 601))
        ground_hist = e.make_kde(np.array(col_ground_pts), 1) if len(col_ground_pts) != 0 else np.zeros((501, 601))
    elif hist_type=="count":
        drop_hist = e.create_count_histogram(drop_pts)
        obs_hist = e.create_count_histogram(col_obs_pts)
        wall_hist = e.create_count_histogram(col_wall_pts)
        ground_hist = e.create_count_histogram(col_ground_pts)
    else:
        raise Exception("Histogram type {} not implemented.".format(hist_type))
    
    return drop_hist, obs_hist, wall_hist, ground_hist

def load_human_heatmap(trial, condition, split="test", cut=None, rt_cleaned=True, hist_type="kde"):

    if cut is None:
        str_add = ""
    else:
        str_add = f"_cut_{cut}"

    if rt_cleaned:
        str_add += "_rt_cleaned"

    if hist_type == "kde":
        hist_type_str = ""
    elif hist_type == "count":
        hist_type_str = "_count"
    else:
        raise Exception(f"Histogram type {hist_type} not implemented.")

    path = f"heatmaps/human_trial_{condition}_{split}{hist_type_str}_{trial}{str_add}.pickle"

    with open(path, "rb") as f:
        human_hm = pickle.load(f)
        
    return human_hm

def load_heatmaps(tr_num, condition, cut=None, hist_type="kde"):

    if hist_type == "kde":
        add_str = ""
    elif hist_type == "count":
        add_str = "_count"

    with open(f"heatmaps/obs_trial{add_str}_{tr_num}.pickle", "rb") as f:
        obs_hm = pickle.load(f)
    with open(f"heatmaps/ball_trial{add_str}_{tr_num}.pickle", "rb") as f:
        ball_hm = pickle.load(f)
    with open(f"heatmaps/holes{add_str}.pickle", "rb") as f:
        hole_hm = pickle.load(f)     
    with open(f"heatmaps/center{add_str}.pickle", "rb") as f:
        center_hm = pickle.load(f)        
    human_hm = load_human_heatmap(tr_num, condition, cut=cut, hist_type=hist_type)

    return obs_hm, ball_hm, hole_hm, center_hm, human_hm

def setup_trial_regression(tr_num, tr_events, model_type, condition, cut=None, hist_type="kde"):
    obs_hm, ball_hm, hole_hm, center_hm, human_hm = load_heatmaps(tr_num, condition, cut=cut, hist_type=hist_type)
    if model_type == "bandit" or model_type == "uniform" or model_type == "mixed" or model_type == "smart" or model_type == "sequential":
        drop_hm, dyn_obs_hm, wall_hm, ground_hm = make_events_heatmaps(tr_events, model_type, hist_type=hist_type)
        if condition == "occluded":
            features = [obs_hm, hole_hm, center_hm, drop_hm, dyn_obs_hm, wall_hm]
        else:
            features = [obs_hm, ball_hm, hole_hm, center_hm, drop_hm, dyn_obs_hm, wall_hm, ground_hm]
    elif model_type == "visual_features":
        if condition == "occluded":
            features = [obs_hm, hole_hm, center_hm]
        else:
            features = [obs_hm, ball_hm, hole_hm, center_hm]
    else:
        raise Exception(f"Model type {model_type} not implemented.")
    
    features = np.array([np.ravel(arr) for arr in features]).T
    labels = np.ravel(human_hm)

    return features, labels

world_files = os.listdir("stimuli/ground_truth/")
world_nums = sorted([int(file[6:-5]) for file in world_files])


def setup_model_regression(model_type, condition, model_events=None, cut=None, hist_type="kde", normalize=False, world_nums=world_nums):
    if model_type == "visual_features":
        model_events = zip(world_nums, [{}]*len(world_nums))
    feature_list, label_list = [], []
    for tr_num, tr_events in model_events:
        if tr_num in world_nums:
            features, part_vec = setup_trial_regression(tr_num, tr_events, model_type, condition, cut=cut, hist_type=hist_type)
            feature_list.append(features)
            label_list.append(part_vec)

    features = np.concatenate(feature_list)
    labels = np.concatenate(label_list)
    if normalize and hist_type == "count":
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        features = (features - feature_min)/(feature_max - feature_min)

        label_min = np.min(labels)
        label_max = np.max(labels)
        labels = (labels - label_min)/(label_max - label_min)

    return features, labels

def compute_regression(model_events, model_type, condition, cut=None, hist_type="kde", normalize=False, world_nums=world_nums):
    model_features, labels = setup_model_regression(model_type, condition, model_events, cut=cut, hist_type=hist_type, normalize=normalize, world_nums=world_nums)
    reg = LinearRegression().fit(model_features, labels)
    model_pred = reg.predict(model_features)

    return reg, model_pred, model_features, labels

def evaluate_model(model_events, model_type):
    _, model_pred, _, labels = compute_regression(model_events, model_type)

    return np.sum((model_pred - labels)**2)