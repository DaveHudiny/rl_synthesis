from tf_agents.policies import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories import StepType

import numpy as np
import tensorflow as tf

from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step

import tensorflow_probability as tfp


from rl_src.tools.encoding_methods import observation_and_action_constraint_splitter

import logging

from rl_src.reward_machines.predicate_automata import PredicateAutomata

logger = logging.getLogger(__name__)

class PolicyMaskWrapper(TFPolicy):
    """Wrapper for stochastic policies that allows to use observation and action constraint splitters"""

    def __init__(self, policy: TFPolicy, observation_and_action_constraint_splitter=observation_and_action_constraint_splitter, 
                 time_step_spec=None, is_greedy : bool = False, select_rand_action_probability : float = 0.0,
                 predicate_automata : PredicateAutomata = None):
        """Initializes the policy mask wrapper, which is a wrapper for stochastic policies which enables to use observation and action constraint splitters.

        Args:
            policy (TFPolicy): Policy, which should be wrapped. This policy does not use masks for action selection.
            observation_and_action_constraint_splitter (func, optional): Splits observations to pure observations and masks. 
                                                                         Defaults to observation_and_action_constraint_splitter from agents.tools.
            time_step_spec (TimeStepSpec, optional): Time Step specification with mask. Defaults to None.
            is_greedy (bool, optional): Whether the policy should be greedy or not. Defaults to False.
        """
        if predicate_automata is not None:
            self.state_spec_len = predicate_automata.get_reward_state_spec_len(predicate_based=False)
            # Combine original spec with the length of the predicate automata visited states vector as a dictionary
            policy_state_spec = policy.policy_state_spec
            if predicate_automata.predicate_based_rewards:
                self.prediate_number = len(predicate_automata.predicate_set_labels)
                policy_state_spec = {**policy_state_spec, 'satisfied_predicates': tf.TensorSpec(shape=(len(predicate_automata.predicate_set_labels), 1), dtype=tf.bool),
                                    "automata_state": tf.TensorSpec(shape=(1,), dtype=tf.int32)}
            else:
                policy_state_spec = {**policy_state_spec, 'visited_automata_states': tf.TensorSpec(shape=(self.state_spec_len, 1), dtype=tf.bool),
                                    "automata_state": tf.TensorSpec(shape=(1,), dtype=tf.int32)}
            self.predicate_automata = predicate_automata
            self.get_initial_automata_state = self.predicate_automata.get_initial_state
            self.step_automata = self.predicate_automata.step
            self.get_initial_visited_states = self._get_initially_visited_states
            if policy.info_spec == ():
                info_spec = {'current_automata_state': tf.TensorSpec(shape=(1,), dtype=tf.int32)}
            else:
                info_spec = {**policy.info_spec, 'current_automata_state': tf.TensorSpec(shape=(1,), dtype=tf.int32)}
        else:
            policy_state_spec = policy.policy_state_spec
            self.predicate_automata = None
            info_spec = policy.info_spec

        super(PolicyMaskWrapper, self).__init__(time_step_spec=time_step_spec,
                                                  action_spec=policy.action_spec,
                                                  policy_state_spec=policy_state_spec,
                                                  info_spec=info_spec,
                                                  observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self._policy = policy
        self._observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self._time_step_spec = time_step_spec
        self._action_spec = policy.action_spec
        self._policy_state_spec = policy_state_spec
        self._info_spec = policy.info_spec
        self._is_greedy = is_greedy
        self._real_distribution = self._distribution
        self._select_random_action_probability = select_rand_action_probability

        self.__policy_masker = lambda logits, mask: tf.compat.v2.where(
            tf.cast(mask, tf.bool), logits, tf.constant(logits.dtype.min, dtype=logits.dtype))
        self.__policy_dummy_masker = lambda logits, mask: logits
        self.current_masker = self.__policy_dummy_masker
        self.epsilon_greedy_probability = 0.0
        self.return_real_logits = False
        self.set_random_selector() # sets the default action selector to random categorical sampling
        

    def epsilon_greedy_action(self, time_step : TimeStep) -> tf.Tensor:
        """Returns a random action from a set of legal actions with a uniform distribution,
        where the legal actions are defined by the mask in the time step observation."""
        _, mask = self._observation_and_action_constraint_splitter(time_step.observation)
        probs = tf.where(
            tf.cast(mask, tf.bool),
            tf.ones_like(mask, dtype=tf.float32) / tf.reduce_sum(tf.cast(mask, tf.float32)),
            tf.zeros_like(mask, dtype=tf.float32)
        )
        # Return a random action from the legal actions with a uniform distribution
        action = tf.random.categorical(
            tf.math.log(probs), num_samples=1, dtype=tf.int32
        )
        action = tf.squeeze(action, axis=-1)
        return action

    def is_greedy(self):
        return self._is_greedy
    
    def set_greedy(self, is_greedy):
        self._is_greedy = is_greedy

    def set_policy_masker(self):
        """Set the policy masker to the default one"""
        self.current_masker = self.__policy_masker
    
    def set_identity_masker(self):
        """Unset the policy masker to the default one"""
        self.current_masker = self.__policy_dummy_masker

    def set_argmax_selector(self):
        """Set the policy to use argmax action selection."""
        self.__action_selector = lambda logits: tf.argmax(logits, output_type=tf.int32, axis=-1)

    def set_random_selector(self):
        """Set the policy to use random action selection."""
        self.__action_selector = lambda logits: tf.random.categorical(logits, num_samples=1, dtype=tf.int32)


    def set_prune_zero_dot_one_probs_selector(self):
        """Set the policy to prune actions with probability less than 0.1 and sample categorical action."""
        def prune_logits(logits):
            probs = tf.nn.softmax(logits, axis=-1)
            pruned_probs = tf.where(probs < 0.1, tf.constant(logits.dtype.min, dtype=logits.dtype), logits)
            normalized_probs = tf.nn.softmax(pruned_probs, axis=-1)
            action = tf.random.categorical(tf.math.log(normalized_probs), num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
            return action
        self.__action_selector = prune_logits
    
    def set_prune_zero_dot_zero_five_probs_selector(self):
        """Set the policy to prune actions with probability less than 0.05 and sample categorical action."""
        def prune_logits(logits):
            probs = tf.nn.softmax(logits, axis=-1)
            pruned_probs = tf.where(probs < 0.05, tf.constant(logits.dtype.min, dtype=logits.dtype), logits)
            normalized_probs = tf.nn.softmax(pruned_probs, axis=-1)
            action = tf.random.categorical(tf.math.log(normalized_probs), num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
            return action
        self.__action_selector = prune_logits

    def set_uniform_top_two_selector(self):
        """Set the policy to select uniformly from the top two actions."""
        def top_two_logits(logits):
            top_two_values, top_two_indices = tf.math.top_k(logits, k=2)
            probs = tf.ones_like(top_two_values, dtype=tf.float32) / 2.0
            action = tf.random.categorical(tf.math.log(probs), num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
            selected_action = tf.gather(top_two_indices, action, batch_dims=1)
            return selected_action
        self.__action_selector = top_two_logits

    def set_uniform_top_three_selector(self):
        """Set the policy to select uniformly from the top three actions."""
        def top_three_logits(logits):
            top_three_values, top_three_indices = tf.math.top_k(logits, k=3)
            probs = tf.ones_like(top_three_values, dtype=tf.float32) / 3.0
            action = tf.random.categorical(tf.math.log(probs), num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
            selected_action = tf.gather(top_three_indices, action, batch_dims=1)
            return selected_action
        self.__action_selector = top_three_logits

    def set_prune_below_zero_dot_one_uniform_selector(self):
        """Set the policy to prune actions with probability less than 0.1 and select uniformly from the rest."""
        def prune_and_uniform_logits(logits):
            probs = tf.nn.softmax(logits, axis=-1)
            pruned_probs = tf.where(probs < 0.1, tf.zeros_like(probs), probs)
            sum_probs = tf.reduce_sum(pruned_probs, axis=-1, keepdims=True)
            normalized_probs = tf.where(sum_probs > 0, pruned_probs / sum_probs, pruned_probs)
            action = tf.random.categorical(tf.math.log(normalized_probs), num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
            return action
        self.__action_selector = prune_and_uniform_logits

    def set_return_real_logits(self, return_real_logits: bool):
        """Set whether the policy should return the real logits or the masked logits."""
        self.return_real_logits = return_real_logits

    def _get_additional_initial_state(self, batch_size):
        state_number = self.get_initial_automata_state()
        state_number = tf.fill((batch_size,), state_number)
        return state_number
    
    def _get_initially_visited_states(self, batch_size):
        # Get the initial state of the automata
        nr_visited_labels = self.state_spec_len
        # Generated zero flag mask
        visited_states = tf.zeros((batch_size, nr_visited_labels), dtype=tf.bool)
        return visited_states

    def generate_curiosity_reward(self, prev_state, next_state, next_policy_step_type):
        # Get the previously_visited_states
        if self.predicate_automata.predicate_based_rewards:
            visited_states = tf.cast(prev_state["satisfied_predicates"], tf.float32)
            next_visited_states = tf.cast(next_state["satisfied_predicates"], tf.float32)
        else:
            visited_states = tf.cast(prev_state["visited_automata_states"], tf.float32)
            next_visited_states = tf.cast(next_state["visited_automata_states"], tf.float32)
        # Get the next visited states
        
        # Compare the visited states, if there is any change for a given batch number, add a reward
        diff = tf.abs(next_visited_states - visited_states)
        # Get the reward
        reward = tf.reduce_max(diff, axis=-1) * 1
        # If there is no new state, give some small penalty
        reward = tf.where(tf.equal(reward, 0.0), -0.05, reward)
        reward = tf.where( # if the next_state is initial state, set the reward to 0
            tf.not_equal(next_policy_step_type, StepType.MID), 0.0, reward
        )
        return reward

    @tf.function
    def _get_initial_state(self, batch_size):
        # print("Getting initial state", batch_size)

        policy_state = self._policy._get_initial_state(batch_size)
        if self.predicate_automata is not None:
            if self.predicate_automata.predicate_based_rewards:
                policy_state["satisfied_predicates"] = tf.zeros((batch_size, self.prediate_number), dtype=tf.bool)
            else:
                policy_state["visited_automata_states"] = self.get_initial_visited_states(batch_size)
            policy_state["automata_state"] = self.get_initial_automata_state(batch_size)
        return policy_state

    def _distribution(self, time_step, policy_state, seed) -> PolicyStep:
        observation, mask = self._observation_and_action_constraint_splitter(
            time_step.observation)
        time_step = time_step._replace(observation=observation)
        distribution = self._policy.distribution(
            time_step, policy_state)
        return distribution

    def _action(self, time_step, policy_state, seed) -> PolicyStep:
        """Returns the action for the given time step and policy state."""
        mask = time_step.observation["mask"]
        distribution = self._real_distribution(time_step, policy_state, seed)

        # If distribution.action is "Deterministic" distribution of tfp, the values are probabilities
        logits = distribution.action.logits
        logits = self.current_masker(logits, mask)
        action = self.__action_selector(logits)
        action = tf.reshape(action, (-1,))
        # if self._is_greedy:
        #     # print("Greedy action")
        #     logits = distribution.action.logits
        #     logits = self.current_masker(logits, mask)
        #     action = tf.argmax(logits, output_type=tf.int32, axis=-1)
        # else:
        #     # print("Stochastic action")
        #     # _, mask = self._observation_and_action_constraint_splitter(time_step.observation)
        #     # action = self._get_action_masked(distribution, mask)
        #     logits = distribution.action.logits
        #     logits = self.current_masker(logits, mask)
        #     # action = tf.random.categorical(distribution.action.logits, num_samples=1, dtype=tf.int32)
        #     action = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=seed)
        #     action = tf.squeeze(action, axis=-1)

        if self.predicate_automata is not None:
            new_policy_state = distribution.state
            # Get the visited states
            if self.predicate_automata.predicate_based_rewards:
                visited_states = tf.cast(policy_state["satisfied_predicates"], tf.bool)
            else:
                visited_states = tf.cast(policy_state["visited_automata_states"], tf.bool)
            # Get the next visited states
            next_visited_state, satisfied_predicates = self.step_automata(policy_state["automata_state"], time_step.observation["observation"])
            new_policy_state["automata_state"] = next_visited_state

            # Create indices from next_visited_state using batch numbers
            if self.predicate_automata.predicate_based_rewards:
                next_visited_state = tf.cast(satisfied_predicates, tf.bool)
                next_visited_state = tf.reshape(next_visited_state, (tf.shape(next_visited_state)[0], -1))
                next_visited_state = tf.logical_or(next_visited_state, visited_states)
                new_policy_state["satisfied_predicates"] = tf.cast(next_visited_state, tf.bool)
            else:
                batch_size = tf.shape(next_visited_state)[0]
                batch_indices = tf.range(batch_size)
                batch_indices = tf.reshape(batch_indices, (batch_size, 1))
                batch_indices = tf.stack([batch_indices, next_visited_state], axis=-1)
                next_state_indices = tf.reshape(batch_indices, (batch_size, 2))
                # Update the visited states with newly observed states
                visited_states = tf.tensor_scatter_nd_update(visited_states, next_state_indices, tf.ones(next_state_indices.shape[0], dtype=tf.bool))

                new_policy_state["visited_automata_states"] = visited_states
            if distribution.info == ():
                info = {'current_automata_state': new_policy_state["automata_state"]}
            else:
                distribution.info["current_automata_state"] = new_policy_state["automata_state"]
                info = distribution.info
        else:
            new_policy_state = distribution.state
            if self.return_real_logits:
                info = {'dist_params': {'logits': logits}}
            else:
                info = distribution.info

        if self.epsilon_greedy_probability > 0.0:
            use_epsilon_greedy = tf.random.uniform((), 0, 1) < self.epsilon_greedy_probability
            action = tf.where(use_epsilon_greedy, self.epsilon_greedy_action(time_step), action)
        policy_step = PolicyStep(action=action, state=new_policy_state, info=info)
        return policy_step