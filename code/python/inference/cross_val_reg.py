import numpy as np
import pandas as pd
import regression_analysis as ra
import evaluate_heatmaps as eh
import pickle
import time
from sys import argv

start_time = time.time()

model_type = argv[1]
task_id = int(argv[2])
num_jobs = int(argv[3])

split_range = np.arange(1, 101, dtype=int)
bwv_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
bws_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# split_range = [1, 2, 3]
# bwv_range = [20, 40, 60]
# bws_range = [0.1, 0.5, 0.9]

with open("train.pkl", "rb") as f:
    train = pickle.load(f)

with open("test.pkl", "rb") as f:
    test = pickle.load(f)

##################
### Procedures ###
##################
def create_filename(model_type, evidence, thr=None, unc=None, sw=None, num_samples=None, start_sim=None, vbw=None, sbw=None, tb=None, tbw=None, beta=None, hole_select="proportional", vision_prior="uniform", audio_prior="uniform", phys_params=(0.3, 0.0, 0.7)):

    dn, cm, cs = phys_params

    filename = f"{model_type}_{evidence}_"

    if model_type == "smart" or model_type == "sequential":
        filename += f"hole_select_{hole_select}_"

        if hole_select == "softmax":
            assert beta is not None
            filename += f"beta_{beta}_"    

    if thr is not None:
        filename += f"runs_30_threshold_{thr}_"

    if unc is not None:
        filename += f"uncertainty_{unc}_"

    if num_samples is not None:
        filename += f"num_samples_{num_samples}_"

    if vbw is not None:
        filename += f"bwv_{vbw}_"

    if sw is not None:
        filename += f"sample_weight_{sw}_"

    if sbw is not None:
        filename += f"bws_{sbw}_"

    if tb is not None:
        filename += f"timing_bins_{tb}_bwt_{tbw}_"

    if start_sim is not None:
        filename += f"start_sim_{start_sim}_"

    if model_type == "bandit" or model_type == "uniform_sampler":
        filename += "prior_vision_"
    elif model_type == "mixed":
        filename += "prior_uniform_"
    elif model_type == "smart" or model_type == "sequential":
        filename += f"prior_{vision_prior}_{audio_prior}_"

    filename += f"phys_params_{dn}_{cm}_{cs}"

    return filename

def fit_train(train_worlds, model_type, evidence, condition, bw_vis=None, bw_sound=None):

    path = "model_performance/collisions/"
    filename = create_filename(model_type, evidence, num_samples=100, vbw=bw_vis, sbw=bw_sound)

    with open(path + filename + ".pkl", "rb") as f:
        model_events = pickle.load(f)

    reg_train, _, _, _ = ra.compute_regression(model_events, model_type, condition, cut=300, world_nums=train_worlds)

    return reg_train, model_events

def predict(reg_train, model_type, model_events, condition):

    features, _ = ra.setup_model_regression(model_type, condition, model_events, cut=300)
    predictions = reg_train.predict(features)

    return predictions

def eval_params(param_list, train):

    for parset in param_list:
        print(f"Evaluating parameters: {parset}")
        print()

        split = parset[0]
        model_type = parset[1]
        evidence = parset[2]

        if evidence == "vision_independent":
            bw_vis = parset[3]
            bw_sound = None
            condition = "vision"
        elif evidence == "vision_sound_independent":
            bw_vis = parset[3]
            bw_sound = parset[4]
            condition = "audio"
        elif evidence == "sound_independent":
            bw_vis = None
            bw_sound = parset[3]
            condition = "occluded"
        else:
            raise ValueError("Unknown evidence type")
        
        train_worlds = train[split-1]

        print("Fitting Regression Model")
        reg_train, model_events = fit_train(train_worlds, model_type, evidence, condition, bw_vis=bw_vis, bw_sound=bw_sound)
        print("Computing predictions")
        predictions = predict(reg_train, model_type, model_events, condition)

        print("Computing EMD")
        emd = eh.compute_emd_all_trials(predictions, condition, cut=300)
        print()

        param_dict = {
            "trial": ra.world_nums,
            "distance": emd,
            "use": ["train" if trial in train_worlds else "test" for trial in ra.world_nums]
        }

        df = pd.DataFrame(param_dict)

        filename = create_filename(model_type, evidence, num_samples=100, vbw=bw_vis, sbw=bw_sound)

        df.to_csv(f"model_performance/cross_val/split{split:03d}/{filename}.csv", index=False)

    return

def create_param_list(split_range, model_type, bwv_range, bws_range):
    param_list = []

    for split in split_range:
        for evidence in ["vision_independent", "vision_sound_independent", "sound_independent"]:
            if evidence == "vision_independent":
                for bw_vis in bwv_range:
                    param_list.append((split, model_type, evidence, bw_vis))
            elif evidence == "vision_sound_independent":
                for bw_vis in bwv_range:
                    for bw_sound in bws_range:
                        param_list.append((split, model_type, evidence, bw_vis, bw_sound))
            elif evidence == "sound_independent":
                for bw_sound in bws_range:
                    param_list.append((split, model_type, evidence, bw_sound))
            else:
                raise Exception(f"Evidence type {evidence} not implemented")

    return param_list

def get_job_params(param_list, num_jobs, job_id):
    """
    job_id is now assumed to be 1-based (i.e., ranges from 1 to num_jobs).
    """
    n = len(param_list)
    # Convert to 0-based index for calculation
    job_idx = job_id - 1
    chunk_size = n // num_jobs
    remainder = n % num_jobs
    start = job_idx * chunk_size + min(job_idx, remainder)
    end = start + chunk_size + (1 if job_idx < remainder else 0)
    return param_list[start:end]

######################
### Main execution ###
######################

param_list = create_param_list(split_range, model_type, bwv_range, bws_range)
job_params = get_job_params(param_list, num_jobs, task_id)
eval_params(job_params, train)
print(f"Task {task_id} completed in {time.time() - start_time:.2f} seconds.")