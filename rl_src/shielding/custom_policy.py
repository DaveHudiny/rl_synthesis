import tensorflow as tf

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep

import numpy as np

from rl_src.tools.encoding_methods import observation_and_action_constraint_splitter
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.environment.tf_py_environment import TFPyEnvironment

import tensorflow_probability as tfp


class CustomPolicy(TFPolicy):
    def __init__(self, action_spec, time_step_spec, policy_state_spec=(), 
                 observation_and_action_constraint_splitter=observation_and_action_constraint_splitter, 
                 distribution_function=None):
        super(CustomPolicy, self).__init__(time_step_spec, action_spec, policy_state_spec, 
                                           observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self._distribution_function = distribution_function

    def _get_initial_state(self, batch_size):
        return super()._get_initial_state(batch_size)

    def _distribution(self, time_step, policy_state):
        observation, mask = self.observation_and_action_constraint_splitter(
            time_step.observation)
        observation = observation.numpy().tolist()
        mask = mask.numpy().tolist()
        probabilities, next_state = self._distribution_function(observation, mask, policy_state)
        probabilities = tf.convert_to_tensor(probabilities, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32) if next_state is not () else policy_state
        # Policy_step with categorical actions
        policy_step = PolicyStep(action=tfp.distributions.Categorical(logits=tf.math.log(probabilities)), state=next_state)
        return policy_step
    
    def _action(self, time_step, policy_state):
        probabilities, next_state = self._distribution(time_step, policy_state)
        probabilities = tf.math.log(probabilities)

        action = tf.random.categorical(probabilities, 1, dtype=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        policy_step = PolicyStep(action=action, state=next_state)
        return policy_step
    
def uniform_random_distribution():
    def distribution_function(observation, mask, policy_state):
        batch_size = len(observation)
        num_actions = len(mask[0])
        probabilities = np.ones((batch_size, num_actions)) / num_actions
        probabilities = probabilities * np.array(mask)
        return probabilities, policy_state
    return distribution_function

def create_uniform_random_policy(environment):

    tf_env = TFPyEnvironment(environment)
    custom_policy = CustomPolicy(
        action_spec=tf_env.action_spec(),
        time_step_spec=tf_env.time_step_spec(),
        distribution_function=uniform_random_distribution()
    )

    return custom_policy

def create_custom_policy(environment, observation_to_action):

    tf_env = TFPyEnvironment(environment)
    custom_policy = CustomPolicy(
        action_spec=tf_env.action_spec(),
        time_step_spec=tf_env.time_step_spec(),
        distribution_function=lambda observation, mask, policy_state: (
            np.array([observation_to_action[tuple(obs)] for obs in observation]), policy_state
        )
    )

    return custom_policy
    
