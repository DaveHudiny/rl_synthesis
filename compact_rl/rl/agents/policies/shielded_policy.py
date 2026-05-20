from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from compact_rl.rl.agents.policies.fsc_copy import FSC

from tf_agents.policies import TFPolicy
from tf_agents.specs.tensor_spec import TensorSpec

import tensorflow as tf

import numpy as np

def observation_and_action_constraint_splitter(observation):
    """Splits the observation into the part used by the policy and the part used for action constraints (masks).
    Args:
        observation: A nest of Tensors representing the observation.
    Returns:
        A tuple (policy_observation, mask, integer_observation).
    """
    policy_observation = observation['observation']
    mask = observation['mask']
    integer_observation = observation['integer']
    return policy_observation, mask, integer_observation

class Shielded_Policy(TFPolicy):
    def __init__(self, time_step_spec, action_spec, original_policy: TFPolicy, shield,
                    observation_and_action_constraint_splitter=None):
        """A TF-Agents policy wrapper that applies shielding to an existing policy.
        Args:
            time_step_spec: A `TimeStep` spec of the environment.
            action_spec: A nest of `TensorSpec` representing the actions.
            original_policy: The original TF-Agents policy to be wrapped.
            shield: The shielding mechanism to apply.
            observation_and_action_constraint_splitter: (Optional) A function that splits
                an observation into the part used by the policy and the part used for
                action constraints (e.g., masks). If None, no splitting is done.
        Returns:
            A new TF-Agents policy that applies shielding.
        """
        policy_state_spec = {
            'original_policy_state': original_policy.policy_state_spec,
            'shield_state': TensorSpec(shape=(), dtype=tf.float32),
        }

        super(Shielded_Policy, self).__init__(
            time_step_spec=time_step_spec,
            policy_state_spec=policy_state_spec,
            action_spec=action_spec,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self.original_policy = original_policy
        self.shield = shield

    def _action(self, time_step, policy_state, seed = None):
        pass


