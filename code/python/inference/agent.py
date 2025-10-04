import copy
import numpy as np
import agent_utils
from scipy.special import factorial, gamma
from scipy.stats import dirichlet_multinomial
import platform
import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

if platform.node().startswith("PSY-"):
	import engine
     


class Agent():
	
    def __init__(self,
                 trial_num,
                 evidence,
                 bw_vis,
                 bw_sound,
                 vision_prior=None,
                 audio_prior=None,
                 agent_vars=None,
                 phys_params=(0.3, 0.0, 0.7),
                 saved_sims=None,
                 experiment="inference"):
        
        """
        Agent class constructor. Sets up the agent's count representations and initializes priors if given.
        Args:
            trial_num (int): The trial number for the agent.
            evidence (str): Type of evidence used by the agent (e.g., "vision_independent", "sound_independent", "vision_sound_independent", "vision_sound_joint").
            bw_vis (float): Bandwidth for vision.
            bw_sound (float): Bandwidth for sound.
            vision_prior (str): Type of prior for vision ("vision", "uniform").
            audio_prior (str): Type of prior for audio ("audio", "uniform").
            agent_vars (list, optional): Preinitialized agent counts and update masks. If None, they will be initialized.
            phys_params (tuple): Physical parameters for the world simulation.
            saved_sims (dict, optional): Precomputed simulations to use instead of running new ones.
            experiment (str): The experiment type, default is "inference".
        """
          
        self.trial_num = trial_num
        drop_noise, col_mean, col_sd = phys_params

        self.world, self.hole_dropped_into, self.n_collisions = agent_utils.get_world(trial_num, drop_noise, col_mean, col_sd, experiment=experiment)
        self.ball_pos = int(np.floor(self.world["ball_final_position_unity"]["x"]))
        self.hole_pos = np.array([x for x,_ in self.world["hole_positions_unity"]], dtype=int)

        self.evidence = evidence
        self.saved_sims = saved_sims

        if agent_vars is None:
             vision_bins = 600
             collision_bins = 6

             agent_vars = agent_utils.initialize_agent_vars(self.evidence,
                                                            vision_bins=vision_bins,
                                                            max_cols=collision_bins,
                                                            n_bins=None,
                                                            bw_vis=bw_vis,
                                                            bw_sound=bw_sound,
                                                            bw_timing=None)
             
        if "independent" in self.evidence:
             
            if "vision" in self.evidence:
                self.counts_vision = copy.deepcopy(agent_vars[0][0])
                self.update_mask_vision = copy.deepcopy(agent_vars[1][0])

                if vision_prior == "vision":
                     self.counts_vision += self.update_mask_vision[self.hole_pos, :]
            
            if "sound" in self.evidence:
                self.counts_sound = copy.deepcopy(agent_vars[0][1])
                self.update_mask_sound = copy.deepcopy(agent_vars[1][1])

                if audio_prior == "audio":
                    assert self.saved_sims is not None, "Saved sims must be provided for audio prior"
                    audio_counts = agent_utils.compute_audio_prior(self.saved_sims)
                    self.counts_sound += audio_counts

        elif "joint" in self.evidence:

            self.counts = copy.deepcopy(agent_vars[0])
            self.update_mask = copy.deepcopy(agent_vars[1])

            if vision_prior == "vision":
                self.counts += self.update_mask[self.hole_pos, 0, :, 0][:,:,np.newaxis]
            if audio_prior != "uniform":
                raise Exception("Joint evidence with audio prior not implemented")

        else:
            raise ValueError("Invalid evidence type: {}".format(self.evidence))
        

    def simulate_world(self, hole, convert_coordinates=True):
        """
        Using the saved agent world, simulate from the given hole.
        Args:
            hole (int): The hole to simulate from.
            convert_coordinates (bool): Whether to convert coordinates to the unity coordinate system.
        """

        if self.saved_sims is None:
            world = copy.deepcopy(self.world)
            world["hole_dropped_into"] = hole
            self.simulated_world = engine.run_simulation(world, convert_coordinates=convert_coordinates, distorted=False)

        else:
            world_hole_sims = self.saved_sims[self.trial_num][hole]
            sim_idx = np.random.choice(len(world_hole_sims))
            self.simulated_world = world_hole_sims[sim_idx]

    def extract_observation(self):

        """
        Extract the observation from the simulated world.
        Returns:
            tuple: The x-coordinate of the ball position and the number of collisions.
        """

        if "vision" in self.evidence:
            ball_pos = int(np.round(self.simulated_world["ball_position"][0]["x"]))
        else:
            ball_pos = None

        if "sound" in self.evidence:
            n_collisions = len(self.simulated_world["collisions"])
        else:
            n_collisions = None
        
        return ball_pos, n_collisions


    def compute_obs_likelihood(self):

        """
        Compute the likelihood of the observation given the agent's counts.
        Returns:
            array: The likelihood of the given observation for the three holes.
        """

        if self.evidence == "vision_independent":
            vision_dist = self.counts_vision/ np.sum(self.counts_vision, axis=1, keepdims=True)
            obs_likelihood = vision_dist[:, self.ball_pos]
        elif self.evidence == "sound_independent":
            sound_dist = self.counts_sound / np.sum(self.counts_sound, axis=1, keepdims=True)
            obs_likelihood = sound_dist[:, self.n_collisions-1]
        elif self.evidence == "vision_sound_independent":
            vision_dist = self.counts_vision / np.sum(self.counts_vision, axis=1, keepdims=True)
            sound_dist = self.counts_sound / np.sum(self.counts_sound, axis=1, keepdims=True)
            obs_likelihood = vision_dist[:, self.ball_pos] * sound_dist[:, self.n_collisions-1]
        elif self.evidence == "vision_sound_joint":
            joint_dist = self.counts / np.sum(self.counts, axis=(1, 2), keepdims=True)
            obs_likelihood = joint_dist[:, self.ball_pos, self.n_collisions-1]
        else:
            raise ValueError("Invalid evidence type: {}".format(self.evidence))
        
        return obs_likelihood
    
    def compute_posterior(self):

        """
        Compute the posterior distribution over the holes given the observation likelihood.
        Args:
            obs_likelihood (array): The likelihood of the observation for each hole.
        Returns:
            array: The posterior distribution over the holes.
        """

        obs_likelihood = self.compute_obs_likelihood()
        posterior = obs_likelihood / np.sum(obs_likelihood)
        return posterior
    
    def update_counts(self, hole, vision_obs=None, sound_obs=None):

        """
        Update the agent's counts based on the observation.
        Args:
            hole (int): The hole the ball was dropped into.
            vision_obs (int, optional): The vision observation (x-coordinate of the ball).
            sound_obs (int, optional): The sound observation (number of collisions).
        """

        if self.evidence == "vision_independent":
            assert vision_obs is not None, "Vision observation must be provided for vision independent evidence"
            self.counts_vision[hole] += self.update_mask_vision[vision_obs]
        elif self.evidence == "sound_independent":
            assert sound_obs is not None, "Sound observation must be provided for sound independent evidence"
            self.counts_sound[hole] += self.update_mask_sound[sound_obs-1]
        elif self.evidence == "vision_sound_independent":
            assert vision_obs is not None and sound_obs is not None, "Both vision and sound observations must be provided for vision_sound_independent evidence"
            self.counts_vision[hole] += self.update_mask_vision[vision_obs]
            self.counts_sound[hole] += self.update_mask_sound[sound_obs-1]
        elif self.evidence == "vision_sound_joint":
            assert vision_obs is not None and sound_obs is not None, "Both vision and sound observations must be provided for vision_sound_joint evidence"
            self.counts[hole, vision_obs, sound_obs-1] += self.update_mask[vision_obs, sound_obs-1]

        else:
            raise ValueError("Invalid evidence type: {}".format(self.evidence))

    

    
        




