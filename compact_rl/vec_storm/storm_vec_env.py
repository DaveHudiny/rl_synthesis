from typing import Set, Dict, Tuple
import json
import pickle

import numpy as np

import jax
from jax import numpy as jnp

from stormpy import simulator

from .sparse_array import SparseArray
from .simulator import Simulator, States, StepInfo, ResetInfo

import logging
logger = logging.getLogger(__name__)


def cast2jax(data):
    if isinstance(data, SparseArray):
        return SparseArray(
            nr_actions = data.nr_actions,
            nr_states = data.nr_states,
            nr_rows = data.nr_rows,
            row_starts = jnp.array(data.row_starts),
            row_ends = jnp.array(data.row_ends),
            indices = jnp.array(data.indices),
            data = jnp.array(data.data)
        )
    if isinstance(data, np.ndarray):
        return jnp.array(data)


class StormVecEnvBuilder:
    NO_LABEL = "__no_label__"

    @classmethod
    def get_action_labels(cls, pomdp, ignore_label) -> Set[str]:
        action_labels = set()
        for state in range(pomdp.nr_states):
            n_act = pomdp.get_nr_available_actions(state)
            for action in range(n_act):
                for label in pomdp.choice_labeling.get_labels_of_choice(pomdp.get_choice_index(state, action)):
                    action_labels.add(label)
        if ignore_label in action_labels:
            action_labels.remove(ignore_label)
        return list(sorted(action_labels))

    @classmethod
    def build_from_pomdp(cls, pomdp, get_scalarized_reward, num_envs=1, seed=42, metalabels=None, max_steps=100, random_init=False,
                         obs_evaluator=None, quotient_state_valuations=None, observation_to_actions=None) -> Simulator:
        """
            pomdp: The POMDP object that should be compiled into a jax-based environment.
            get_scalarized_reward: A function that accepts a dictionary indexed by reward signal names and returns a number.
        """

        sim = simulator.create_simulator(pomdp)
        action_labels = cls.get_action_labels(pomdp, cls.NO_LABEL)
        action_labels2indices = {label: i for i, label in enumerate(action_labels)}
        initial_state = pomdp.initial_states[0]

        rewards_types = sim.get_reward_names()
        nr_states = pomdp.nr_states
        cls.nr_states = nr_states
        nr_actions = len(action_labels)
        # Row map: Assigns each (state, action) pair a row in the spare transition/reward matrix
        # or -1 if the action is not allowed in the state
        # DH added: state_values_by_ids to enable work on state estimators combining full and partial observability.
        logger.info("Computing row map")
        row_map = np.zeros(nr_states * nr_actions, dtype=np.int32)
        row_map[:] = -1
        sinks = np.zeros(nr_states, dtype=bool)
        if not hasattr(pomdp, "observations"):
            states_to_observations = np.arange(nr_states)
        else:
            states_to_observations = np.array(pomdp.observations)

        try:
            state_valuations = pomdp.state_valuations
        except:
            state_valuations = quotient_state_valuations

        
        state_labels = list(json.loads(str(state_valuations.get_json(0))).keys())
        nr_state_labes = len(state_labels)
        state_values_by_ids = np.zeros((pomdp.nr_states, nr_state_labes), dtype=np.float32)
        for state in range(nr_states):
            sinks[state] = pomdp.is_sink_state(state)
            for action_offset in range(pomdp.get_nr_available_actions(state)):
                labels = pomdp.choice_labeling.get_labels_of_choice(pomdp.get_choice_index(state, action_offset))
                for label in labels:
                    if label != cls.NO_LABEL:
                        action_idx = action_labels2indices[label]
                        row_map[state * nr_actions + action_idx] = pomdp.transition_matrix.get_rows_for_group(state)[action_offset]
            state_valuation_json = json.loads(str(state_valuations.get_json(state)))
            state_values_by_ids[state] = np.array(list(state_valuation_json.values()), dtype=np.float32)

        # Transitions
        logger.info("Computing transitions")
        transitions = SparseArray.from_data(nr_states, nr_actions, row_map, pomdp.transition_matrix)
        # Allowed actions
        logger.info("Computing allowed actions")
        allowed_actions = (row_map != -1).reshape(nr_states, nr_actions)

        # if observation_to_actions is not None: # Observation to actions contains mapping [ [legal_action1, legal_action2, ...], [legal_action1], ...] 
        #     allowed_actions = np.zeros((nr_states, nr_actions), dtype=bool)
        #     states_converted_to_observations = states_to_observations[np.arange(nr_states)]
        #     for state in np.arange(states_converted_to_observations.shape[0]):
        #         obs_id = states_converted_to_observations[state]
        #         legal_actions = observation_to_actions[obs_id]
        #         allowed_actions[state, legal_actions] = True
            
            

        # Sinks
        logger.info("Computing sinks")
        
        # sinks = ~allowed_actions.any(axis=-1)

        # Raw rewards
        logger.info("Computing raw rewards")
        raw_rewards = {}

        for reward_type in rewards_types:
            raw_rewards[reward_type] = SparseArray.zeros_like(transitions)
            reward_model = pomdp.reward_models[reward_type]
            if reward_model.has_state_rewards:
                rewards = np.array(reward_model.state_rewards)
                for sa_idx in range(nr_states * nr_actions):
                    next_states = raw_rewards[reward_type].get_row_indices_np(sa_idx)
                    raw_rewards[reward_type].get_row_np(sa_idx)[:] += rewards[next_states]

            if reward_model.has_transition_rewards:
                raise NotImplementedError("Transition rewards are not supported")

            if reward_model.has_state_action_rewards:
                state_action_rewards = np.array(reward_model.state_action_rewards)
                for sa_idx in range(nr_states * nr_actions):
                    row_idx = row_map[sa_idx]
                    if row_idx == -1:
                        continue
                    raw_rewards[reward_type].get_row_np(sa_idx)[:] += state_action_rewards[row_idx]

        # Assign labels to states
        logger.info("Computing labels")
        labeling = pomdp.labeling
        labels = {}
        for label in labeling.get_labels():
            labels[label] = np.zeros(nr_states, dtype=bool)
            for state in labeling.get_states(label):
                labels[label][state] = 1

        # Rewards
        logger.info("Computing scalarized rewards")
        rewards = SparseArray.zeros_like(transitions)
        reward_data = {
            reward_type: raw_rewards[reward_type].data
            for reward_type in rewards_types
        }
        rewards.data = get_scalarized_reward(reward_data, rewards_types)

        # Metalabels
        logger.info("Computing metalabels")
        
        metalabel_keys = None if metalabels is None else list(metalabels.keys())
        
        metalabels_data = None
        if metalabel_keys is not None:
            metalabels_data = np.ones((nr_states, len(metalabels)), dtype=bool)
            for i, m in enumerate(metalabel_keys):
                for l in metalabels[m]:
                    metalabels_data[:, i] &= labels[l]
        else:
            metalabels_data = np.zeros((nr_states, 0), dtype=bool)

        # Observations
        logger.info("Computing observations")

        if hasattr(pomdp, "observations"):
            try:
                valuations = pomdp.observation_valuations
                nr_observables = len(json.loads(str(valuations.get_json(0))))

                observations_by_ids = np.zeros((pomdp.nr_observations, nr_observables), dtype=np.float32)
                observation_labels = list(json.loads(str(valuations.get_json(0))).keys())

                for obs_id in range(pomdp.nr_observations):
                    valuation_json = json.loads(str(valuations.get_json(obs_id)))
                    observations_by_ids[obs_id] = np.array(list(valuation_json.values()), dtype=np.float32)
                state_observation_ids = states_to_observations
                observations = observations_by_ids[state_observation_ids]
            except:
                if obs_evaluator is None:
                    raise ValueError("POMDP has observations, but no observation valuations and no obs_evaluator provided.")
                observation_labels = obs_evaluator.obs_expr_label
                valuator = obs_evaluator.obs_valuation
                nr_observables = len(observation_labels)
                observations_by_ids = np.zeros((pomdp.nr_observations, nr_observables), dtype=np.float32)
                for obs_id in range(pomdp.nr_observations):
                    observations_by_ids[obs_id] = np.array(valuator(obs_id), dtype=np.float32)
                state_observation_ids = states_to_observations
                observations = observations_by_ids[pomdp.observations]
        else: # Full observability. Use the state values as observations
            observations_by_ids = state_values_by_ids
            observation_labels = state_labels
            state_observation_ids = np.arange(nr_states)
            observations = state_values_by_ids

        # Save simulator data
        return Simulator(
            id = Simulator.get_free_id(),
            initial_state = initial_state,
            max_outcomes = (transitions.row_ends-transitions.row_starts).max(),
            max_steps = max_steps,
            random_init = random_init,
            transitions = cast2jax(transitions),
            rewards = cast2jax(rewards),
            observations = cast2jax(observations),
            sinks = cast2jax(sinks),
            allowed_actions = cast2jax(allowed_actions),
            labels = labels,
            metalabels = cast2jax(metalabels_data),
            action_labels = action_labels,
            observation_labels = observation_labels,
            state_values = cast2jax(state_values_by_ids),
            state_labels = state_labels,
            state_observation_ids = cast2jax(state_observation_ids),
            observation_by_ids = cast2jax(observations_by_ids),
        )


class StormVecEnv:
    """
        Class that provides a fast sparse vectorized representation of a Storm environment.
        It uses JAX to compile the topology extracted from the given model, thus accelerating the interactions.
    """

    def __init__(self, pomdp, get_scalarized_reward: Dict[str, np.array], num_envs=1, seed=42, metalabels=None, random_init=False, max_steps=100,
                 obs_evaluator=None, quotient_state_valuations=None, observation_to_actions=None):
        """
            pomdp: The POMDP object that should be compiled into a jax-based environment.
            get_scalarized_reward: A function that accepts a dictionary indexed by reward signal names and returns an array of scalarized rewards.

            Example:
            get_scalarized_reward = lambda rewards: rewards['reward1'] + rewards['reward2']

            num_envs: The number of environments to be simulated in parallel.
            seed: The seed for the random number generator.
            metalabels: A dictionary of metalabels to be used in the environment. For each metalabel,
            the dictionary should contain a list of labels whose conjunction defines the metalabel.

            Example:
            metalabels = {
                "metalabel1": ["label1", "label2"],
                "metalabel2": ["label3"]
            }

            labels(s1) = [label1, label2] => metalabel1(s1) = True
            labels(s2) = [label3] => metalabel2(s2) = True
            labels(s3) = [label1] => metalabel1(s3) = False
        """
        self.simulator_states = States(
            vertices = jnp.zeros(num_envs, jnp.int32),
            steps = jnp.zeros(num_envs, jnp.int32),
        )
        self.rng_key = jax.random.key(seed)
        self.random_init = random_init
        self.simulator = StormVecEnvBuilder.build_from_pomdp(
            pomdp,
            get_scalarized_reward,
            num_envs=num_envs,
            seed=seed,
            metalabels=metalabels,
            max_steps=max_steps,
            random_init=random_init,
            obs_evaluator=obs_evaluator,
            quotient_state_valuations=quotient_state_valuations,
            observation_to_actions=observation_to_actions
        )

    def enable_random_init(self):
        self.simulator.id = Simulator.get_free_id()
        self.simulator.random_init = True
        jax.clear_caches()
    
    def disable_random_init(self):
        self.simulator.id = Simulator.get_free_id()
        self.simulator.random_init = False
        jax.clear_caches()
    
    def set_num_envs(self, num_envs):
        self.simulator_states = States(
            vertices = jnp.zeros(num_envs, jnp.int32),
            steps = jnp.zeros(num_envs, jnp.int32),
        )
        jax.clear_caches()
    
    def set_seed(self, seed):
        self.rng_key = jax.random.key(seed)

    def reset(self) -> Tuple[jnp.array, jnp.array, jnp.array]:
        """
            Reset all the environments. If `random_init` is set to True, the initial state will be randomly selected.

            Returns:
            - observations: The observations of the new states.
            - allowed_actions: The boolean mask of allowed actions for the new states.
            - metalabels: The boolean mask of metalabels for the new states.
        """
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        res: ResetInfo = self.simulator.reset(self.simulator_states, reset_key)
        self.simulator_states = res.states
        self.simulator_integer_observations = res.integer_observations
        return res.observations, res.allowed_actions, res.metalabels
    
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
        res: StepInfo = self.simulator.step(self.simulator_states, actions, step_key)
        self.simulator_states = res.states
        self.simulator_integer_observations = res.integer_observations
        return res.observations, res.rewards, res.done, res.truncated, res.allowed_actions, res.metalabels

    def get_label(self, label, vertices=None):
        if vertices is None:
            return self.get_label(label, self.simulator_states.vertices)
    
        return self.simulator.labels[label][vertices]

    def get_labels(self, vertices=None):
        if vertices is None:
            return self.get_labels(self.simulator_states.vertices)
        
        return {
            key: val[vertices] for key, val in self.simulator.labels.items()
        }

    def get_action_labels(self):
        """
            Get list of action labels that occur in the environment.
        """
        return self.simulator.action_labels

    def get_observation_labels(self):
        """
            Get list of observation labels that occur in the environment.
        """
        return self.simulator.observation_labels
    
    def get_state_labels(self):
        return self.simulator.state_labels
    
    def get_state_values(self):
        return self.simulator.state_values

    def save(self, file: str):
        pickle.dump(self, open(file, "wb"))
    
    @classmethod
    def load(cls, file: str):
        env = pickle.load(open(file, "rb" ))
        env.simulator.id = Simulator.get_free_id()
        return env

    @property
    def nr_states(self):
        return len(self.simulator.sinks)
    
    def __del__(self):
        try:
            jax.clear_caches()
        except:
            pass

    def set_states(self, states):
        """
            Set the states of the environments. This is useful for testing policies, jumpstarts, Go-Explore or other purposes like debugging.
            states: The states to be set.
        """
        self.simulator_states = States(
            vertices = jnp.array(states, jnp.int32),
            steps = jnp.zeros(len(states), jnp.int32), # TODO: Perhaps this should be better to set for a non-zero value, if we want to simulate only a couple of steps.
        )
        self.simulator.states = self.simulator_states

        self.simulator_integer_observations = self.simulator.observations[self.simulator_states.vertices]

    def no_step(self) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """
            Perform a step in the environment without taking any action. This is used for obtaining the current simulator state.
            Returns:
            - observations: The observations of the new states (after the potential reset).
            - rewards: The rewards of the transitions (before the potential reset).
            - done: A boolean mask indicating if the new states are terminal (before the potential reset).
            - truncated: A boolean mask indicating if the new states reached the maximum number of steps (before the potential reset).
            - allowed_actions: The boolean mask of allowed actions for the new states (after the potential reset).
            - metalabels: The boolean mask of metalabels for the new states (before the potential reset).
        """
        res : StepInfo = self.simulator.no_step(self.simulator_states)
        return res.observations, res.rewards, res.done, res.truncated, res.allowed_actions, res.metalabels
