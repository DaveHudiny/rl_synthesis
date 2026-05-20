import tensorflow as tf

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep

import numpy as np

from compact_rl.rl.tools.encoding_methods import observation_and_action_constraint_splitter

from compact_rl.rl.tests.general_test_tools import init_args, load_sketch
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
from compact_rl.rl.tools.evaluators import evaluate_policy_in_model

import tensorflow_probability as tfp

def create_dummy_distribution():
    def distribution_function(observation, mask, policy_state):
        batch_size = len(observation)
        num_actions = len(mask[0])
        probabilities = np.ones((batch_size, num_actions)) / num_actions
        probabilities = probabilities * np.array(mask)
        return probabilities, policy_state
    return distribution_function

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
    
    def _action(self, time_step, policy_state, seed):
        policy_step = self._distribution(time_step, policy_state)
        probabilities = tf.math.log(policy_step.action.logits)

        action = tf.random.categorical(probabilities, 1, dtype=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        policy_step = PolicyStep(action=action, state=policy_step.state)
        return policy_step
    
def test_custom_policy():
    project_path = "models/models_pomdp_no_family/network-3-8-20"

    args = init_args(prism_path=None, properties_path=None,
                     use_rnn_less=False, # Use RNN-less agent (if True, the policy should be completely memoryless)
                     max_steps=601, # Max steps per episode
                     seed=42, # Random seed, for the reproducibility, set it to some integer value
                     prefer_stochastic=True, # Whether to prefer stochastic or deterministic actions during the evaluation
                    )
    sketch = load_sketch(project_path=project_path)
    model = sketch.pomdp # If you don't have POMDP, you can switch to quotient mdp or some other MDP/POMDP representations.
    # model = sketch.quotient_mdp

    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True)

    tf_env = TFPyEnvironment(environment)
    custom_policy = CustomPolicy(
        action_spec=tf_env.action_spec(),
        time_step_spec=tf_env.time_step_spec(),
        distribution_function=create_dummy_distribution()
    )
    # If you use TF functions, the custom policy must be compatible with TF graph execution.
    evaluate_policy_in_model(custom_policy, args, environment, tf_env, use_tf_function=False)
    