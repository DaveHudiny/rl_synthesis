from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec
import numpy as np

from interpreters.extracted_fsc.extracted_fsc_policy import ExtractedFSCPolicy

from tests.general_test_tools import init_args, init_environment
from agents.recurrent_ppo_agent import Recurrent_PPO_Agent
from tools.evaluators import evaluate_policy_in_model

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

class TableBasedPolicy(TFPolicy):
    def __init__(self, original_policy : TFPolicy, action_function : np.ndarray, 
                 update_function : np.ndarray, # Action and update function has shape (nr_model_states, nr_observations) 
                 initial_memory = 0,
                 action_keywords = None,
                 descending_actions = None,
                 nr_observations = None,
                 time_step_spec : TimeStep = None,
                 action_spec = None
                 ):
        """
        TableBasedPolicy is a policy that uses a table to map observations to actions and updates.
        
        Args:
            original_policy (TFPolicy): The original policy to be used. (TODO: Remove this, since you can derive the policy from the model.)
            action_function (np.ndarray): The action function table. It has shape (nr_model_states, nr_observations).
            update_function (np.ndarray): The update function table. It has shape (nr_model_states, nr_observations)."""
        policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)

        if time_step_spec is not None and action_spec is not None:
            super(TableBasedPolicy, self).__init__(time_step_spec, action_spec, policy_state_spec=policy_state_spec)
        else:
            super(TableBasedPolicy, self).__init__(original_policy.time_step_spec, original_policy.action_spec, policy_state_spec=policy_state_spec)
        self.nr_actions = len(action_keywords)
        self.nr_observations = nr_observations if nr_observations is not None else action_function.shape[1]
        self.mem_size = action_function.shape[0]
        self.joint_transition_function = None
        if (action_function is not None) and (update_function is None): # Only one of the functions is provided, and thus the action and update depends on each other
            self.is_joint = True
            self.joint_transition_function = action_function if action_function is not None else update_function # shape (nr_model_states, nr_observations, nr_actions * nr_model_states)
            self.joint_transition_function = tf.constant(self.joint_transition_function, dtype=tf.float32)
            self.fix_joint_transition_probs()
        elif action_function is not None and update_function is not None: # Both functions are provided, and thus the action and update are independent
            self.is_joint = False
            self.tf_observation_to_action_table = tf.constant(action_function, dtype=tf.float32)
            self.tf_observation_to_update_table = tf.constant(update_function, dtype=tf.float32)
            self.fix_action_table_probs()
            if descending_actions is not None: # This array contains actions for a given memory and observation in descending order
                                           # The first action is the most likely one, but it could be illegal in some cases
                                           # This array is used to get the most likely legal action, when combined mask
                self.descending_actions = tf.constant(descending_actions, dtype=tf.int32)
            else:
                self.descending_actions = None

        else:
            raise ValueError("Both action_function and update_function cannot be None.")

        self.action_keywords = action_keywords
        self.initial_memory = initial_memory

    def fix_joint_transition_probs(self):
        if self.nr_observations is not None and self.nr_observations != self.joint_transition_function.shape[1]:
            # Fill the joint transition function with zeros for the missing observations
            missing_observations = self.nr_observations - self.joint_transition_function.shape[1]
            if missing_observations > 0:
                zeros = tf.zeros((self.joint_transition_function.shape[0], missing_observations, self.joint_transition_function.shape[-1]), dtype=tf.float32)
                self.joint_transition_function = tf.concat([self.joint_transition_function, zeros], axis=1)
        if self.joint_transition_function.shape[-1] != self.nr_actions * self.joint_transition_function.shape[0]:
            # Fill the joint transition function with zeros for the missing actions
            missing_actions = self.joint_transition_function.shape[-1] - self.nr_actions * self.joint_transition_function.shape[0]
            if missing_actions > 0:
                zeros = tf.zeros((self.joint_transition_function.shape[0], self.joint_transition_function.shape[1], missing_actions), dtype=tf.float32)
                self.joint_transition_function = tf.concat([self.joint_transition_function, zeros], axis=-1)
        # Normalize the joint transition function to ensure that it is a valid probability distribution
        normalizers = tf.reduce_sum(self.joint_transition_function, axis=-1, keepdims=True)
        self.joint_transition_function = tf.math.divide_no_nan(self.joint_transition_function, normalizers)
        # Repair the joint transition function to ensure that it is a valid probability distribution
        self.joint_transition_function = tf.where(normalizers > 0,
                                                  self.joint_transition_function, 
                                                  tf.ones_like(self.joint_transition_function) / tf.cast(tf.shape(self.joint_transition_function)[-1], 
                                                                                                          dtype=tf.float32))

    def fix_action_table_probs(self):
        """Fixes the action table probabilities to sum to 1 for each observation."""
        if self.nr_observations is not None and self.nr_observations != self.tf_observation_to_action_table.shape[1]:
            # Fill the action table with zeros for the missing observations
            missing_observations = self.nr_observations - self.tf_observation_to_action_table.shape[1]
            if missing_observations > 0:
                zeros = tf.zeros((self.tf_observation_to_action_table.shape[0], missing_observations, self.tf_observation_to_action_table.shape[-1]), dtype=tf.float32)
                self.tf_observation_to_action_table = tf.concat([self.tf_observation_to_action_table, zeros], axis=1)
        if self.tf_observation_to_action_table.shape[-1] != self.tf_observation_to_update_table.shape[-1]:
            # Fill the update table with zeros for the missing updates
            missing_updates = self.tf_observation_to_action_table.shape[1] - self.tf_observation_to_update_table.shape[1]
            if missing_updates > 0:
                zeros = tf.zeros((self.tf_observation_to_update_table.shape[0], missing_updates, self.tf_observation_to_update_table.shape[2]), dtype=tf.float32)
                self.tf_observation_to_update_table = tf.concat([self.tf_observation_to_update_table, zeros], axis=1)
        if self.tf_observation_to_action_table.shape[-1] > 1:
            normalizers = tf.reduce_sum(self.tf_observation_to_action_table, axis=-1, keepdims=True)
            self.tf_observation_to_action_table = tf.math.divide_no_nan(self.tf_observation_to_action_table, normalizers)
            # Repair the action table to ensure that it is a valid probability distribution
            self.tf_observation_to_action_table = tf.where(normalizers > 0, 
                                                           self.tf_observation_to_action_table, 
                                                           tf.ones_like(self.tf_observation_to_action_table) / tf.cast(tf.shape(self.tf_observation_to_action_table)[-1], 
                                                                                                                       dtype=tf.float32))
        if self.tf_observation_to_update_table.shape[-1] > 1:
            normalizers = tf.reduce_sum(self.tf_observation_to_update_table, axis=-1, keepdims=True)
            self.tf_observation_to_update_table = tf.math.divide_no_nan(self.tf_observation_to_update_table, normalizers)
            # Repair the update table to ensure that it is a valid probability distribution
            self.tf_observation_to_update_table = tf.where(normalizers > 0, 
                                                           self.tf_observation_to_update_table, 
                                                           tf.ones_like(self.tf_observation_to_update_table) / tf.cast(tf.shape(self.tf_observation_to_update_table)[-1], 
                                                                                                                       dtype=tf.float32))

    def _get_initial_state(self, batch_size):
        return tf.constant(self.initial_memory, shape=(batch_size, 1), dtype=tf.int32)
    
    def _action_independents(self, observation, mask, policy_state):
        memory = policy_state
        indices = tf.concat([memory, observation], axis=1)
        if self.descending_actions is not None:
            # Gather the ordered actions for the current observation, remove the illegal actions and select the first legal action
            descending_actions = tf.gather_nd(self.descending_actions, indices)
            descending_actions = tf.boolean_mask(descending_actions, mask)
            action = descending_actions[0]
        else:
            action = tf.gather_nd(self.tf_observation_to_action_table, indices)
        # If actions are arrays of probabilities, we need to sample from them
        if isinstance(action, tf.Tensor) and len(action.shape) > 1:
            action = tf.random.categorical(tf.math.log(action), num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
        else:
            action = tf.cast(action, dtype=tf.int32)
        update = tf.gather_nd(self.tf_observation_to_update_table, indices)
        
        if isinstance(update, tf.Tensor) and len(update.shape) > 1:
            # Get the indices of the maximum update value for each observation. These indices correspond to the selected update.
            update = tf.random.categorical(tf.math.log(update), num_samples=1, dtype=tf.int32)
            update = tf.reshape(update, (-1, 1))
        else:
            update = tf.reshape(update, (-1, 1))
            update = tf.cast(update, dtype=tf.int32)
        policy_step = PolicyStep(action, update)
        return policy_step

    def _joint_action(self, observation, mask, policy_state):
        """Joint action function that uses the joint transition function."""
        memory = policy_state
        indices = tf.concat([memory, observation], axis=1)
        joint_transition_probs = tf.gather_nd(self.joint_transition_function, indices)
        # Sample from the joint transition probabilities
        joint_action = tf.random.categorical(tf.math.log(joint_transition_probs), num_samples=1, dtype=tf.int32)
        joint_action = tf.squeeze(joint_action, axis=-1)
        action = joint_action // self.mem_size
        update = joint_action % self.mem_size
        action = tf.cast(action, dtype=tf.int32)
        action = tf.reshape(action, (-1,))
        update = tf.cast(update, dtype=tf.int32)
        update = tf.reshape(update, (-1, 1))
        policy_step = PolicyStep(action, update)
        return policy_step


    @tf.function
    def _action(self, time_step, policy_state, seed):
        observation = time_step.observation["integer"]
        mask = time_step.observation["mask"]
        if not self.is_joint:
            policy_step = self._action_independents(observation, mask, policy_state)
        else:
            policy_step = self._joint_action(observation, mask, policy_state)
        return policy_step
    
def initialize_random_independent_fsc_function(action_keywords, nr_observations, num_fsc_states):
    action_function = np.random.uniform(0.1, 1.0, (num_fsc_states, nr_observations, len(action_keywords)))
    update_function = np.random.uniform(0.1, 1.0, (num_fsc_states, nr_observations, num_fsc_states))
    action_function = action_function / np.sum(action_function, axis=-1, keepdims=True) 
    update_function = update_function / np.sum(update_function, axis=-1, keepdims=True)
    return action_function, update_function

def initialize_random_joint_fsc_function(action_keywords, nr_observations, num_fsc_states):
    """Initialize a random joint FSC."""
    action_function = np.random.uniform(0.1, 1.0, (num_fsc_states, nr_observations, num_fsc_states * len(action_keywords)))
    action_function = action_function.reshape((num_fsc_states, nr_observations, num_fsc_states * len(action_keywords)))
    action_function = action_function / np.sum(action_function, axis=-1, keepdims=True) 

    return action_function, None
    
if __name__ == "__main__":
    args = init_args("models/evade/sketch.templ", "models/evade/sketch.props")

    environment, tf_env = init_environment(args)
    action_keywords = environment.action_keywords
    nr_observations = environment.stormpy_model.nr_observations
    num_fsc_states = 3

    # action_function, update_function = initialize_random_independent_fsc(action_keywords, nr_observations, num_fsc_states)

    recurrent_ppo = Recurrent_PPO_Agent(environment, tf_env, args)
    action_function, update_function = initialize_random_joint_fsc_function(action_keywords, nr_observations, num_fsc_states)

    policy = TableBasedPolicy(
        original_policy=recurrent_ppo.get_policy(False, True),
        action_function=action_function,
        update_function=update_function,
        initial_memory=0,
        action_keywords=action_keywords,
        descending_actions=None
    )
    eager = PyTFEagerPolicy(policy, use_tf_function=True) # Makes the execution faster, but is not necessary for the policy to work

    evaluate_policy_in_model(policy, args, environment, tf_env, evaluation_steps=400)



