import os
from tqdm import tqdm

import numpy as np
import os
import pandas as pd

import gzip
import pickle
import warnings
warnings.simplefilter("ignore")

import agent_utils
from agent import Agent

def run_uniform(trial_num,
                evidence,
                num_samples,
                bw_vis,
                bw_sound,
                agent_vars=None,
                phys_params=(0.3, 0.0, 0.7),
                saved_sims=None):
    
    """
    Run the uniform sampler for a given trial.
    Args:
        trial_num (int): The trial number.
        evidence (str): Type of evidence used by the agent.
        num_samples (int): Number of samples to draw for each hole.
        bw_vis (float): Bandwidth for vision.
        bw_sound (float): Bandwidth for sound.
        agent_vars (dict, optional): Precomputed agent variables to use instead of running new ones.
        phys_params (tuple): Physical parameters for the world simulation.
        saved_sims (dict, optional): Precomputed simulations to use instead of running new ones.
    """

    agent = Agent(trial_num=trial_num,
                  evidence=evidence,
                  bw_vis=bw_vis,
                  bw_sound=bw_sound,
                  agent_vars=agent_vars,
                  phys_params=phys_params,
                  saved_sims=saved_sims)
    
    collision_record = {"drop": [], "col_obs": [], "col_wall": [], "col_ground": []}

    for hole in [0, 1, 2]:

        for _ in range(num_samples):

            agent.simulate_world(hole)
            sim_ball_pos, sim_ncol = agent.extract_observation()
            agent.update_counts(hole, vision_obs=sim_ball_pos, sound_obs=sim_ncol)

            drop_pt = agent.simulated_world['drop']['pos']
            collision_record['drop'].append([drop_pt['x'], drop_pt['y']])

            for col in agent.simulated_world['collisions']:
                col_point = col["look_point"]
                x = col_point['x']
                y = col_point['y']
                col_type = col["objects"][1]
                if col_type in ["triangle", "rectangle", "pentagon"]:
                    collision_record['col_obs'].append([x, y])
                elif col_type == "ground":
                    collision_record['col_ground'].append([x, y])
                elif col_type == "walls":
                    collision_record['col_wall'].append([x, y])
                else:
                    raise ValueError(f"Unknown collision type: {col_type}")

    posterior = agent.compute_posterior()

    return posterior, collision_record


def run_uniform_all_trials(param,
                           precompute_sims=False,
                           DATA_PATH="model_performance/"):
    
    """
    Run the uniform sampler for all trials.
    Args:
        param (tuple): Parameters for the uniform sampler.
        precompute_sims (bool): Whether to precompute simulations.
        DATA_PATH (str): Path to save the data.
    """

    dn = param[-3]
    cm = param[-2]
    csd = param[-1]

    if precompute_sims:
        with gzip.open(f"saved_sims/saved_sims_sum_nsims_1000_drop_noise_{dn}_col_mean_{cm}_col_sd_{csd}.gz", "rb") as f:
            saved_sims = pickle.load(f)
    else:
        saved_sims = None

    evidence = param[0]
    num_samples = param[1]
    par_ind = 2

    if "vision" in evidence:
        bw_vis = param[par_ind]
        par_ind += 1
    else:
        bw_vis = None

    if "sound" in evidence:
        bw_sound = param[par_ind]
        par_ind += 1
    else:
        bw_sound = None

    agent_vars = agent_utils.initialize_agent_vars(evidence,
                                                   vision_bins=600,
                                                   max_cols=6,
                                                   n_bins=None,
                                                   bw_vis=bw_vis,
                                                   bw_sound=bw_sound,
                                                   bw_timing=None)
    
    world_num_list = os.listdir("stimuli/ground_truth/")
    world_num_list = sorted([int(x[6:-5]) for x in world_num_list])

    judgment_dict = {"trial": [], "judgment": [], "num_cols": []}
    collisions = []

    for tr_num in tqdm(world_num_list):
        posterior, collision_record = run_uniform(trial_num=tr_num,
                                                  evidence=evidence,
                                                  num_samples=num_samples,
                                                  bw_vis=bw_vis,
                                                  bw_sound=bw_sound,
                                                  agent_vars=agent_vars,
                                                  phys_params=(dn, cm, csd),
                                                  saved_sims=saved_sims)

        num_cols = len(collision_record['col_obs']) + len(collision_record['col_wall']) + len(collision_record['col_ground'])
        judgment_dict["trial"].append(tr_num)
        judgment_dict["judgment"].append(list(posterior))
        judgment_dict["num_cols"].append(num_cols)
        collisions.append((tr_num, collision_record))

    df_judgment = pd.DataFrame(judgment_dict)

    judgment_filename = DATA_PATH + f"judgment_rt/uniform_{evidence}_num_samples_{num_samples}"

    if "vision" in evidence:
        judgment_filename += f"_bwv_{bw_vis}"
    if "sound" in evidence:
        judgment_filename += f"_bws_{bw_sound}"
    judgment_filename += f"_phys_params_{dn}_{cm}_{csd}.csv"

    collision_filename = judgment_filename.replace("judgment_rt", "collisions")
    collision_filename = collision_filename.replace("csv", "pkl")

    df_judgment.to_csv(judgment_filename, index=False)

    with open(collision_filename, "wb") as f:
        pickle.dump(collisions, f)

    return df_judgment, collisions

    