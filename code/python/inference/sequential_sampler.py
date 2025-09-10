import numpy as np
import pandas as pd
from tqdm import tqdm
import agent_utils
from agent import Agent
import pickle
import gzip
import os

def softmax(x, beta=1):
    """
    Compute the softmax of a vector x with temperature beta.
    Args:
        x (array-like): Input vector.
        beta (float): Temperature parameter.
    Returns:
        array: Softmax probabilities.
    """
    e_x = np.exp(beta * x)
    return e_x / e_x.sum(axis=0)

def run_seq(trial_num,
            num_samples,
            evidence,
            bw_vis,
            bw_sound,
            beta=1,
            saved_sims=None,
            hole_select="proportional",
            vision_prior="uniform",
            audio_prior="uniform",
            agent_vars=None,
            phys_params=(0.3, 0.0, 0.7)):
    
    """
    Run the sequential sampler for a given trial.
    Args:
        trial_num (int): The trial number.
        num_samples (int): Number of samples to draw for each hole.
        evidence (str): Type of evidence used by the agent.
        bw_vis (float): Bandwidth for vision.
        bw_sound (float): Bandwidth for sound.
        beta (float): Temperature parameter for softmax.
        saved_sims (dict, optional): Precomputed simulations to use instead of running new ones.
        hole_select (str): Method for selecting holes ("proportional", "softmax", "hardmax").
        vision_prior (str): Type of prior for vision ("vision", "uniform").
        audio_prior (str): Type of prior for audio ("audio", "uniform").
        granularity_params (tuple): Parameters for granularity in vision and collision bins.
        phys_params (tuple): Physical parameters for the world simulation.
    """

    agent = Agent(trial_num=trial_num,
                  evidence=evidence,
                  bw_vis=bw_vis,
                  bw_sound=bw_sound,
                  vision_prior=vision_prior,
                  audio_prior=audio_prior,
                  agent_vars=agent_vars,
                  phys_params=phys_params,
                  saved_sims=saved_sims)
    
    collision_record = {"drop": [], "col_obs": [], "col_wall": [], "col_ground": []}
    hole_count = [0, 0, 0]

    for _ in range(num_samples*3):

        posterior = agent.compute_posterior()

        if hole_select == "proportional":
            hole = np.random.choice([0, 1, 2], p=posterior)
        elif hole_select == "softmax":
            hole_probs = softmax(posterior, beta=beta)
            hole = np.random.choice([0, 1, 2], p=hole_probs)
        elif hole_select == "hardmax":
            hole = np.argmax(posterior)
        else:
            raise ValueError("Invalid hole selection method. Choose 'proportional', 'softmax', or 'hardmax'.")
        
        hole_count[hole] += 1

        agent.simulate_world(hole)
        sim_ball_pos, sim_ncol = agent.extract_observation()
        agent.update_counts(hole, vision_obs=sim_ball_pos, sound_obs=sim_ncol)

        drop_pt = agent.simulated_world['drop']['pos']
        collision_record['drop'].append([drop_pt['x'], drop_pt['y']])
        for col in agent.simulated_world['collisions']:
            col_point = col['look_point']
            x = col_point['x']
            y = col_point['y']
            col_type = col['objects'][1]
            if col_type in ['triangle', 'rectangle', 'pentagon']:
                collision_record['col_obs'].append([x, y])
            elif col_type == "ground":
                collision_record['col_ground'].append([x, y])
            elif col_type == "walls":
                collision_record['col_wall'].append([x, y])
            else:
                raise ValueError(f"Unknown collision type: {col_type}")

    posterior = agent.compute_posterior()

    return posterior, collision_record, hole_count

def run_seq_all_trials(param,
                       precompute_sims=True,
                       DATA_PATH="model_performance/"):
    
    """
    Run the sequential sampler for all trials. Save performance to file.
    Args:
        param (dict): Parameters for the sampler.
        precompute_sims (bool): Whether to precompute simulations.
        DATA_PATH (str): Path to save the results.
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
    hole_select = param[1]

    if hole_select == "softmax":
        beta = param[2]
        par_ind = 3
    else:
        beta = None
        par_ind = 2

    num_samples = param[par_ind]
    par_ind += 1

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

    vision_prior = param[par_ind]
    par_ind += 1

    audio_prior = param[par_ind]
    par_ind += 1

    start = 0
    end = 150

    world_num_list = os.listdir("stimuli/ground_truth/")
    world_num_list = sorted([int(x[6:-5]) for x in world_num_list])
    world_num_list = world_num_list[start:end]

    agent_vars = agent_utils.initialize_agent_vars(evidence,
                                                  vision_bins=600,
                                                  max_cols=6,
                                                  n_bins=None,
                                                  bw_vis=bw_vis,
                                                  bw_sound=bw_sound,
                                                  bw_timing=None)
    
    judgment_dict = {"trial": [], "judgment": [], "num_cols": []}
    collisions = []

    for tr_num in tqdm(world_num_list):

        posterior, collision_record, _ = run_seq(trial_num=tr_num,
                                                 num_samples=num_samples,
                                                 evidence=evidence,
                                                 bw_vis=bw_vis,
                                                 bw_sound=bw_sound,
                                                 beta=beta,
                                                 saved_sims=saved_sims,
                                                 hole_select=hole_select,
                                                 vision_prior=vision_prior,
                                                 audio_prior=audio_prior,
                                                 agent_vars=agent_vars,
                                                 phys_params=(dn, cm, csd))
        
        posterior = list(posterior)
        num_cols = len(collision_record['col_obs']) + len(collision_record['col_wall']) + len(collision_record['col_ground'])
        judgment_dict["trial"].append(tr_num)
        judgment_dict["judgment"].append(posterior)
        judgment_dict["num_cols"].append(num_cols)
        collisions.append((tr_num, collision_record))

    df_judgment = pd.DataFrame(judgment_dict)

    judgment_filename = DATA_PATH + f"judgment_rt/sequential_{evidence}_hole_select_{hole_select}"

    if hole_select == "softmax":
        judgment_filename += f"_beta_{beta}"

    judgment_filename += f"_num_samples_{num_samples}"

    if "vision" in evidence:
        judgment_filename += f"_bwv_{bw_vis}"
    if "sound" in evidence:
        judgment_filename += f"_bws_{bw_sound}"

    judgment_filename += f"_prior_{vision_prior}_{audio_prior}"
    judgment_filename += f"_phys_params_{dn}_{cm}_{csd}.csv"

    collision_filename = judgment_filename.replace("judgment_rt", "collisions")
    collision_filename = collision_filename.replace(".csv", ".pkl")

    df_judgment.to_csv(judgment_filename, index=False)

    with open(collision_filename, "wb") as f:
        pickle.dump(collisions, f)

    return df_judgment, collisions