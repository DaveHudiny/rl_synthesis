from vec_storm import StormVecEnv
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from tf_agents.policies.tf_policy import TFPolicy
import tensorflow as tf

def extract_memory_less_fsc_actions(environment : EnvironmentWrapperVec, policy: TFPolicy, get_probs : bool = False):
    nr_obs = environment.vectorized_simulator.simulator.observation_by_ids.shape[0]
    fake_observations = environment.create_fake_timestep_from_observation_integer(range(nr_obs))
    initial_state = policy.get_initial_state(batch_size=nr_obs)
    if get_probs == False:
        action_step = policy.action(fake_observations, initial_state)
        actions = action_step.action.numpy()
    else:
        action_step = policy.distribution(fake_observations, initial_state)
        logits = action_step.action.logits
        actions = tf.nn.softmax(logits, axis=-1).numpy()
    return actions
    