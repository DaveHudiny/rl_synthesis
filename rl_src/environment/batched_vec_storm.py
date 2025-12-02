from vec_storm.storm_vec_env import StormVecEnv
from vec_storm.storm_vec_env import StepInfo

from typing import Set, Dict, Tuple
import json
import pickle

import numpy as np

import jax
from jax import numpy as jnp

from tools.args_emulator import ArgsEmulator

import os


class BatchedVecStorm(StormVecEnv):
    """
    A vectorized environment for running multiple instances of a VecStorm simulator in parallel.
    This class extends the StormVecEnv to mainain the same interface while allowing for batch processing.
    Initially, it is just the StormVecEnv with a different name, but you can add more POMDPs and then the simulator will play them in parallel.
    """
    def __init__(self, pompd, get_scalarized_reward: Dict[str, np.array], num_envs=1, seed=42, metalabels=None, random_init=False, max_steps=100,
                 obs_evaluator=None, quotient_state_valuations=None, observation_to_actions=None, args: ArgsEmulator = None):
        super().__init__(pomdp, get_scalarized_reward, num_envs=num_envs, seed=seed, metalabels=metalabels,
                         random_init=random_init, max_steps=max_steps, obs_evaluator=obs_evaluator,
                         quotient_state_valuations=quotient_state_valuations,
                         observation_to_actions=observation_to_actions)
        self.get_scalarized_reward = get_scalarized_reward
        self.num_envs = num_envs
        self.metalabels = metalabels
        self.random_init = random_init
        self.max_steps = max_steps
        self.obs_evaluator = obs_evaluator
        self.quotient_state_valuations = quotient_state_valuations
        self.observation_to_actions = observation_to_actions
        self.simulators = [self.simulator] # Initial simulator from the superclass
        self.num_envs_per_simulator = [self.num_envs]  # Track number of environments per simulator

    
    def set_args(self, args: ArgsEmulator):
        """Sets the arguments for the environment."""
        self.args = args
        
    def recompute_num_envs(self, exponential_simulator_distribution=True):
        """Recomputes the number of environments per simulator based on the current state of the simulators."""
        if hasattr(self, "args") and self.args is not None:
            exponential_simulator_distribution = self.args.geometric_batched_vec_storm
        num_simulators = len(self.simulators)
        print(f"Recomputing number of environments per simulator: {num_simulators} simulators, total {self.num_envs} environments.")
        print(f"Exponential distribution: {exponential_simulator_distribution}")
        exponent = 0.4  # Exponential distribution factor
        assert num_simulators > 0, "No simulators available."
        if exponential_simulator_distribution:
            a1 = self.num_envs * (1 - exponent)
            a1 = a1 / (1 - exponent ** num_simulators)
            self.num_envs_per_simulator_reversed = [int(a1 * (exponent ** i)) for i in range(num_simulators)]
            self.num_envs_per_simulator = self.num_envs_per_simulator_reversed[::-1]
            # Ensure the total number of environments matches the original number
            total_envs = sum(self.num_envs_per_simulator)
            if total_envs < self.num_envs:
                # If the total is less, distribute the remaining environments evenly
                for i in range(self.num_envs - total_envs):
                    self.num_envs_per_simulator[-(i % num_simulators)] += 1
        else:
            self.num_envs_per_simulator = [self.num_envs // num_simulators] * num_simulators
            # Distribute any remaining environments evenly across simulators
            for i in range(self.num_envs % num_simulators):
                self.num_envs_per_simulator[i] += 1
        print(f"Number of environments per simulator: {self.num_envs_per_simulator}, total: {sum(self.num_envs_per_simulator)}")

    def set_dominant_simulator(self, simulator_index):
        """Sets the dominant simulator to be used for the next reset and steps.

        Args:
            simulator_index (int): The index of the simulator to be set as dominant.
        """
        if 0 <= simulator_index < len(self.simulators):
            self.num_envs_per_simulator = [0] * len(self.simulators)
            self.num_envs_per_simulator[simulator_index] = self.num_envs
        else:
            raise ValueError(f"Simulator index {simulator_index} is out of bounds. Must be between 0 and {len(self.simulators) - 1}.")
        self.reset()


    def add_pomdp(self, pomdp):
        """Creates new simulator for the given POMDP and adds it to the batch of simulators.

        Args:
            pomdp (SparsePomdp): The POMDP to be added to the batch of simulators.
        """
        _, new_key = jax.random.split(self.rng_key)
        new_key = int(sum(jax.random.key_data(new_key)))  # Convert JAX key to an integer for reproducibility
        new_storm_vec_env = StormVecEnv(
            pomdp=pomdp,
            get_scalarized_reward=self.get_scalarized_reward,
            num_envs=self.num_envs // (len(self.simulators) + 1),  # Distribute environments evenly 
            metalabels=self.metalabels,
            random_init=self.random_init,
            max_steps=self.max_steps,
            obs_evaluator=self.obs_evaluator,
            quotient_state_valuations=self.quotient_state_valuations,
            observation_to_actions=self.observation_to_actions,
            seed=new_key
        )

        new_simulator = new_storm_vec_env.simulator

        self.simulators.append(new_simulator)
        if hasattr(self, "args") and self.args is not None:
            self.recompute_num_envs(self.args.geometric_batched_vec_storm)
        else:
            self.recompute_num_envs()
        self.reset() # Reset to initialize the new simulator states

    def set_num_envs(self, num_envs):
        super().set_num_envs(num_envs)
        self.num_envs = num_envs
        if hasattr(self, "args") and self.args is not None:
            self.recompute_num_envs(self.args.geometric_batched_vec_storm)
        else:
            self.recompute_num_envs()
        self.reset()

    def step(self, actions) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """
            Perform a step in the environment.

            actions: The actions to be taken in the current states.

            Returns:
            - observations: The observations of the new states (after the potential reset).
            - rewards: The rewards of the transitions (before the potential reset).
            - done: A boolean mask indicating if the new states are terminal (before the potential reset).
            - truncated: A boolean mask indicating if the new states reached the maximum number of steps (before the potential reset).
            - allowed_actions: The boolean mask of allowed actions for the new states (after the potential reset).
            - metalabels: The boolean mask of metalabels for the new states (before the potential reset).
        """
        self.rng_key, step_key = jax.random.split(self.rng_key)
        
        # res: StepInfo = self.simulator.step(self.simulator_states, actions, step_key)
        # Call step on each simulator in the batch
        res_list = []
        for i, simulator in enumerate(self.simulators):
            num_envs = self.num_envs_per_simulator[i]
            num_prev_envs = sum(self.num_envs_per_simulator[:i]) if i > 0 else 0
            if num_envs == 0: # If the number of environments for this simulator is zero, skip it.
                continue
            actions_for_simulator = actions[num_prev_envs:num_prev_envs + num_envs]
            res: StepInfo = simulator.step(self.simulator_states.slice(num_prev_envs, num_prev_envs + num_envs), actions_for_simulator, step_key)
            res_list.append(res)

        # Combine results from all simulators
        res = StepInfo.combine(res_list)
        self.simulator_states = res.states
        self.simulator_integer_observations = res.integer_observations
        return res.observations, res.rewards, res.done, res.truncated, res.allowed_actions, res.metalabels



        


        
    