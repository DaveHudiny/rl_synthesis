from vec_storm import StormVecEnv
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from tf_agents.policies.tf_policy import TFPolicy
import tensorflow as tf

def extract_memory_less_fsc_actions(environment : EnvironmentWrapperVec, policy: TFPolicy, get_probs : bool = False, compile_policy : bool = False):

    if compile_policy:
        policy_function = tf.function(policy.distribution)
    else:
        policy_function = policy.distribution

    nr_obs = environment.vectorized_simulator.simulator.observation_by_ids.shape[0]
    fake_observations = environment.create_fake_timestep_from_observation_integer(range(nr_obs))
    initial_state = policy.get_initial_state(batch_size=nr_obs)
    if get_probs == False:
        action_step = policy_function(fake_observations, initial_state)
        logits = action_step.action.logits
        actions = tf.one_hot(tf.argmax(logits, axis=1), depth=tf.shape(logits)[1], dtype=tf.float32).numpy()
    else:
        action_step = policy_function(fake_observations, initial_state)
        logits = action_step.action.logits
        actions = tf.nn.softmax(logits).numpy()
    obs_list = fake_observations.observation['observation'].numpy().tolist()
    obs_tuples = [tuple(obs) for obs in obs_list]
    observation_to_action = dict(zip(obs_tuples, actions.tolist()))
    return actions, observation_to_action
    