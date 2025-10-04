import numpy as np
from scipy.stats import beta, multivariate_normal, norm
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

from utils import load_trial
from convert_coordinate import convertCoordinate

def initialize_counts(granularity_params=(600,6,2), model_version="vision_independent"):
    """
    Args:
        granularity_params: tuple, (vision_bins, collision_bins)
        model_version: str, model version
    """

    if "joint" in model_version:

        if "vision_sound_timing" in model_version:
            counts = np.ones((3, granularity_params[0], granularity_params[2]))
        elif "vision_sound" in model_version:
            counts = np.ones((3, granularity_params[0], granularity_params[1]))
        else:
            raise Exception(f"Model version {model_version} not recognized for joint inference")
        
        return counts
    
    elif "independent" in model_version:

        if "vision_sound_timing" in model_version:
            counts_vision = np.ones((3, granularity_params[0]))
            counts_sound = np.ones((3, granularity_params[1]))
            counts_timing = np.ones((3, granularity_params[1], granularity_params[2]))

        elif "sound_timing" in model_version:
            counts_vision = None
            counts_sound = np.ones((3, granularity_params[1]))
            counts_timing = np.ones((3, granularity_params[1], granularity_params[2]))

        elif "vision_sound" in model_version:
            counts_vision = np.ones((3, granularity_params[0]))
            counts_sound = np.ones((3, granularity_params[1]))
            counts_timing = None

        elif "vision" in model_version:
            counts_vision = np.ones((3, granularity_params[0]))
            counts_sound = None
            counts_timing = None

        elif "sound" in model_version:
            counts_vision = None
            counts_sound = np.ones((3, granularity_params[1]))
            counts_timing = None

        else:
            raise Exception(f"Model version {model_version} not recognized for independent inference")

        return counts_vision, counts_sound, counts_timing
    
    else:
        raise Exception("Model must use either joint or independent representation")
    

def create_masks(inf_type, bins, bws):

    """
    Create the update masks for the agent's counts.

    Args:
        inf_type (str): the type of distribution to use to compute the mask spread.
        bins (tuple): the number of bins for vision and collision.
        bws (tuple): the bandwidths for vision and collision.

    Returns:
        np.ndarray: the computed update masks.
    """

    if inf_type == "multi_normal":

        vision_bins, collision_bins = bins
        bw_vision, bw_collision = bws
        # Create an empty array to store the masks
        masks = np.empty((vision_bins, collision_bins, vision_bins, collision_bins))

        # Create a grid of coordinates
        x, y = np.mgrid[0:vision_bins, 0:collision_bins]

        # Iterate over each point in the grid
        for i in range(vision_bins):
            for j in range(collision_bins):
                # Create a 2D Gaussian mask centered at (i, j)
                rv = multivariate_normal([i, j], [[bw_vision**2, 0], [0, bw_collision**2]])
                mask = rv.pdf(np.dstack((x, y)))

                # Normalize the mask so that it scales from 1 at the center point down to 0
                mask /= mask.max()

                # Store the mask
                masks[i, j] = mask

        return masks
    
    elif inf_type == "uni_normal":

        masks = np.zeros((bins, bins))
        for i in range(bins):
            for j in range(i+1):
                masks[i, j] = np.exp(-0.5 * ((i - j) / bws) ** 2)
            for j in range(i+1, bins):
                masks[i, j] = np.exp(-0.5 * ((j - i) / bws) ** 2)

        return masks
    
    elif inf_type == "uni_geom":

        masks = np.zeros((bins, bins))
        for i in range(bins):
            for j in range(i+1):
                masks[i, j] = bws ** (i - j)
            for j in range(i+1, bins):
                masks[i, j] = bws ** (j - i)

        return masks
    
    elif inf_type == "uni_delta":
        
        masks = np.zeros((bins, bins))
        for i in range(bins):
            masks[i, i] = 1

        return masks
    
    elif inf_type == "multi_delta":
        
        vision_bins, collision_bins = bins
        masks = np.zeros((vision_bins, collision_bins, vision_bins, collision_bins))
        for i in range(vision_bins):
            for j in range(collision_bins):
                masks[i, j, i, j] = 1

        return masks
    
    elif inf_type == "mixed_delta_normal":

        vision_bins, collision_bins = bins
        bw_vision, bw_collision = bws
        masks = np.zeros((vision_bins, collision_bins, vision_bins, collision_bins))
        
        for i in range(vision_bins):
            for j in range(collision_bins):
                if bw_collision == 0:
                    mask = norm.pdf(np.arange(vision_bins), i, bw_vision)
                    mask = mask / mask.sum()
                    masks[i, j, :, j] = mask
                elif bw_vision == 0:
                    mask = norm.pdf(np.arange(collision_bins), j, bw_collision)
                    mask = mask / mask.sum()
                    masks[i, j, i, :] = mask
                else:
                    raise ValueError("Only one of vision_bins or collision_bins can be 0")
                
        return masks
    
    else:
        raise Exception("Inference type not recognized")
    



def get_update_mask(granularity_params=(600,6,10),
                    bws=(10,1,2),
                    model_version="vision_independent"):
    """
    Args:
        granularity_params: tuple, (vision_bins, collision_bins)
        bws: tuple, (bw_vision, bw_collision)
        model_version: str, model version
    """

    vision_bins, collision_bins, timing_bins = granularity_params
    bw_vision, bw_collision, bw_timing = bws

    if "joint" in model_version:

        if "vision_sound_timing" in model_version:

            return create_masks("uni_normal", vision_bins, bw_vision)

        elif "vision_sound" in model_version:

            if bw_vision == 0 and bw_collision == 0:
                inf_type = "multi_delta"
            elif bw_vision == 0 or bw_collision == 0:
                inf_type = "mixed_delta_normal"
            else:
                inf_type = "multi_normal"

            return create_masks(inf_type, (vision_bins, collision_bins), (bw_vision, bw_collision))
        
        else:
            raise Exception(f"Model version {model_version} not recognized for joint inference")
    
    elif "independent" in model_version:

        if bw_vision == 0:
            vision_inf_type = "uni_delta"
        else:
            vision_inf_type = "uni_normal"

        if bw_collision == 0:
            collision_inf_type = "uni_delta"
        else:
            collision_inf_type = "uni_normal"


        if "vision_sound_timing" in model_version:
            vision_masks = create_masks(vision_inf_type, vision_bins, bw_vision)
            collision_masks = create_masks(collision_inf_type, collision_bins, bw_collision)
            timing_masks = create_masks("uni_normal", timing_bins, bw_timing)

        elif "sound_timing" in model_version:
            vision_masks = None
            collision_masks = create_masks(collision_inf_type, collision_bins, bw_collision)
            timing_masks = create_masks("uni_normal", timing_bins, bw_timing)

        elif "vision_sound" in model_version:

            vision_masks = create_masks(vision_inf_type, vision_bins, bw_vision)
            collision_masks = create_masks(collision_inf_type, collision_bins, bw_collision)
            timing_masks = None

        elif "vision" in model_version:

            vision_masks = create_masks(vision_inf_type, vision_bins, bw_vision)
            collision_masks = None
            timing_masks = None

        elif "sound" in model_version:
            vision_masks = None
            collision_masks = create_masks(collision_inf_type, collision_bins, bw_collision)
            timing_masks = None

        else:
            raise Exception(f"Model version {model_version} not recognized for independent inference")

        return vision_masks, collision_masks, timing_masks
    
    else:
        raise Exception("Model must use either joint or independent representation")
        


def get_multinomial_weights(geom_decay=0.1, max_collision=10):
    """
    Get the weights for the multinomial distribution
    :param geom_decay: the geometric decay rate
    :param max_collision: the maximum number of collisions
    """
    update_alpha = np.zeros((max_collision, max_collision))
    for i in range(max_collision):
        for j in range(i+1):
            update_alpha[i, j] = geom_decay ** (i - j)
        for j in range(i+1, max_collision):
            update_alpha[i, j] = geom_decay ** (j - i)
    return update_alpha

# re-implement get_multinomial_weights with a gaussian shape instead of geometric
def get_multinomial_weights_gaussian(bw, bins=520):
    """
    Get the weights for the multinomial distribution
    :param bw: bandwidth of the gaussian
    :param bins: the number of bins
    """
    update_alpha = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(i+1):
            update_alpha[i, j] = np.exp(-0.5 * ((i - j) / bw) ** 2)
        for j in range(i+1, bins):
            update_alpha[i, j] = np.exp(-0.5 * ((j - i) / bw) ** 2)
    return update_alpha


def get_world(trial_num, 
	          drop_noise=0.2, 
	          col_mean=0.8, 
	          col_sd=0.2, 
	          experiment="inference", 
	          hole=None):
    """
    Args:
        trial_num: int
        drop_noise: float
        col_mean: float
        col_sd: float
        experiment: str
	    hole: bool
	"""
    world = load_trial(trial_num, drop_noise=drop_noise, col_mean=col_mean, col_sd=col_sd, experiment=experiment, hole=hole)
    world['ball_final_position_unity'] = {'x': convertCoordinate(world['ball_final_position']['x'], world['ball_final_position']['y'])[0],  'y': convertCoordinate(world['ball_final_position']['x'], world['ball_final_position']['y'])[1]}
    world['hole_positions_unity'] = [convertCoordinate(hole_pos, 600) for hole_pos in world['hole_positions']]
    world["obstacle_positions_unity"] = []
    for _, obs_obj in world['obstacles'].items():
        world['obstacle_positions_unity'].append(convertCoordinate(obs_obj['position']['x'], obs_obj['position']['y']))
    # world['obstacle_positions_unity'] = [convertCoordinate(x, 600)[0] for x in [world['obstacles'][pos]['position']['x'] for pos in world['obstacles']]]

    hole_dropped_into = world['hole_dropped_into']
    n_collisions = len(world['simulation'][hole_dropped_into]['collisions'])
    world['n_collisions'] = n_collisions
    return world, hole_dropped_into, n_collisions


def get_bins(x_min=40, 
             x_max=560, 
             step_min=1, 
             step_max=200, 
             num_bins=(520, 10),
             ):
    """
    Args:
        x_min: float, minimum x value (after correcting for frame size)
        x_max: float, maximum x value (after correcting for frame size)
        step_min: float, minimum step value (i.e. possibls collisions sounds start here)
        step_max: float, maximum step value (i.e. possibls collisions sounds end here)
        num_bins: int, number of bins (i.e. granularity of discretization; default is 10)
    """
    bins_vision = np.linspace(x_min + (x_max - x_min) / num_bins[0], x_max, num_bins[0])
    bins_step = np.linspace(step_min + (step_max - step_min) / num_bins[1], step_max, num_bins[1])
    return bins_vision, bins_step

def initialize_agent_vars(model_version,
                          vision_bins,
                          max_cols,
                          n_bins,
                          bw_vis,
                          bw_sound,
                          bw_timing):
    
    counts = initialize_counts(granularity_params=(vision_bins, max_cols, n_bins), model_version=model_version)
    update_mask = get_update_mask(granularity_params=(vision_bins, max_cols, n_bins), bws=(bw_vis, bw_sound, bw_timing), model_version=model_version)

    return counts, update_mask


def compute_audio_prior(saved_sims):

    """
    Compute the initial audio counts for the structured audio prior.
    Args:
        saved_sims (dict): The saved simulations from which to compute the audio prior.
    Output:
        audio_counts (np.ndarray): The computed audio counts for the structured audio prior.
    """

    world_num_list = list(saved_sims.keys())
     
    audio_counts = np.zeros((3, 6), dtype=int)
    for world_num in world_num_list:
        for hole in range(3):
            for sim in range(1000):
                num_col = len(saved_sims[world_num][hole][sim]["collisions"])
                audio_counts[hole, num_col-1] += 1

    audio_counts = audio_counts/np.max(audio_counts, axis=1, keepdims=True)

    return audio_counts


def generate_timing_obs(n_collisions, n_bins):
    if n_bins == 1:
        yield np.array([n_collisions])
    else:
        for i in range(n_collisions + 1):
            for t in generate_timing_obs(n_collisions - i, n_bins - 1):
                yield np.concatenate(([i], t))


def get_timing_bins(num_bins=6, saved_sims=None):

    if saved_sims is None:
        with open("saved_sims/saved_sims_sum_nsims_1000_drop_noise_0.2_col_mean_0.1_col_sd_0.6.pkl", "rb") as f:
            saved_sims = pickle.load(f)

    col_time = []
    for _, holes in saved_sims.items():
        for hole in holes:
            for sim in hole:
                for col in sim["collisions"]:
                    col_time.append(col["step"])

    bin_boundaries = np.quantile(col_time, np.linspace(0, 1, num_bins+1))

    # Ensure that the last bin boundary is one greater than the maximum value
    bin_boundaries[-1] += 1

    return bin_boundaries[1:]

def get_timing_obs_set(n_bins, max_col=6):

    obs_set = []

    for ncol in range(1, max_col+1):
        col_obs = np.stack(list(generate_timing_obs(ncol, n_bins)))
        obs_set.append(col_obs)

    return obs_set

def convert_col_bin_list(col_bin_list, num_bins):

    obs_key = [0]*num_bins

    for bin in col_bin_list:
        obs_key[bin] += 1

    return np.array(obs_key)

def timing_setup(num_bins, max_col=6, saved_sims=None):

    obs_set = get_timing_obs_set(num_bins, max_col)
    # timing_obs_dict = {bin: i for i , bin in enumerate(timing_obs)}
    bin_boundaries = get_timing_bins(num_bins, saved_sims)

    return obs_set, bin_boundaries


def init_obstacle_check_sound(world, hole, n_collisions, hole_offset=20):
    """
    Args:
        world: world info
        hole: int, hole number
        n_collisions: int, number of collisions
    """
    obstacle_penalty = 1
    hole_x = world['hole_positions_unity'][hole]
    obstacle_count = 0
    for obstacle in world['obstacle_positions_unity']:
        if (hole_x - hole_offset) <= obstacle <= (hole_x + hole_offset):
            obstacle_count += 1
    if obstacle_count == 0 and n_collisions > 1:
        obstacle_penalty = 0
    elif obstacle_count > 0 and n_collisions == 1:
        obstacle_penalty = 0
    return obstacle_penalty

def init_obstacle_check_vision(world, hole, ball_x, hole_offset=20):
    """
    Args:
        world: world info
        hole: int, hole number
        ball_x: float, ball x position
    """
    obstacle_penalty = 0
    hole_x = world['hole_positions_unity'][hole]
    obstacle_count = 0
    for obstacle in world['obstacle_positions_unity']:
        if (hole_x - hole_offset) <= obstacle <= (hole_x + hole_offset):
            obstacle_count += 1
    if obstacle_count == 0:
        if (hole_x - hole_offset) <= ball_x <= (hole_x + hole_offset):
            obstacle_penalty = 1
        elif (hole_x - hole_offset) > ball_x or ball_x > (hole_x + hole_offset):
            obstacle_penalty = 0
    return obstacle_penalty
