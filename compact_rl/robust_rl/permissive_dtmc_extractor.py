from tf_agents.policies import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment


import numpy as np
 

class Permissive_DTMC_Extractor:
    def __init__(self, model, policy : TFPolicy, environment : EnvironmentWrapperVec):
        self.model = model
        self.policy = policy
        self.environment = environment

    def create_replay_buffer(self, num_steps=1000, nr_envrionments=256) -> TFUniformReplayBuffer:
        replay_buffer = TFUniformReplayBuffer(
            data_spec=self.policy.trajectory_spec,
            batch_size=nr_envrionments,
            max_length=num_steps)
        return replay_buffer

    def create_driver(self, replay_buffer : TFUniformReplayBuffer, num_steps=1000, nr_environments=256) -> DynamicStepDriver:
        self.tf_env = TFPyEnvironment(self.environment)
        driver = DynamicStepDriver(
            self.tf_env,
            self.policy,
            observers=[replay_buffer.add_batch],
            num_steps=num_steps * nr_environments)
        return driver

    def create_permissive_policy(self, replay_buffer : TFUniformReplayBuffer, threshold=0.1) -> np.ndarray:
        """
            Checks all the integer values of observations (observation['integer']) and actions pairs in the replay buffer
        and creates a statistic of played actions for each observation. The permissive policy is then constructed by
        including all actions for an observation that have been played with a probability greater than the threshold."""

        
        dataset = replay_buffer.gather_all()

        observations = dataset.observation["integer"].numpy()
        observations = np.reshape(observations, (observations.shape[0], -1))
        actions = dataset.action.numpy()
        actions = np.reshape(actions, (actions.shape[0], -1))

        nr_observations = self.model.nr_observations
        nr_actions = self.environment.nr_actions

        observation_action_counts = np.zeros((nr_observations, nr_actions), dtype=np.int32)

        np.add.at(observation_action_counts, (observations, actions), 1)

        # For observations with sum of actions zero, allow all actions that are allowed in the environment. 
        observations_with_no_actions = np.where(observation_action_counts.sum(axis=-1) == 0)[0]
        allowed_actions_set = self.environment.get_allowed_actions_for_observations(np.arange(nr_observations).tolist())
        for obs in observations_with_no_actions:
            observation_action_counts[obs, allowed_actions_set[obs]] = 1

        policy = observation_action_counts / observation_action_counts.sum(axis=-1, keepdims=True)
        policy = np.nan_to_num(policy)
        policy = np.where(policy > threshold, policy, 0.0)

        # Normalise the policy again
        policy = policy / policy.sum(axis=-1, keepdims=True)

        
        return policy

    def make_permissive_policy_uniform(self, policy):
        """
        Makes the permissive policy uniform by setting all non-zero probabilities to 1.0 and normalizing the policy.
        Args:
            policy (np.ndarray): A 2D array representing the permissive policy, where each row corresponds to an observation and each column corresponds to an action.
        Returns:
            uniform_policy (np.ndarray): A 2D array representing the uniform permissive policy.
        """
        uniform_policy = np.where(policy > 0, 1.0, 0.0)
        uniform_policy = uniform_policy / uniform_policy.sum(axis=-1, keepdims=True)
        return uniform_policy
        

    def construct_permissive_dtmc(self, permissive_policy):
        """
        Constructs a permissive DTMC from the given permissive policy.
        Args:
            permissive_policy (np.ndarray): A 2D array representing the permissive policy, where each row corresponds to an observation and each column corresponds to an action.
        Returns:
            dtmc (stormpy.DTMC): A DTMC constructed from the permissive policy.
        """
        pass

    def extract_dtmc(self, num_steps=1000, nr_environments=256, threshold=0.1) -> None:
        """
        Extracts a DTMC from the given policy and environment by simulating the policy in the environment.
        Args:
            num_steps (int): Number of steps to simulate in each environment for extraction.
            nr_environments (int): Number of parallel environments to use for extraction.
            threshold (float): Threshold for determining the permissiveness of the extracted DTMC (i.e., the minimum probability for an action to be included in the DTMC).
        """
        replay_buffer = self.create_replay_buffer(num_steps, nr_environments)
        driver = self.create_driver(replay_buffer, num_steps, nr_environments)
        driver.run()
        policy = self.create_permissive_policy(replay_buffer, threshold)
        uniform_policy = self.make_permissive_policy_uniform(policy)
        print(uniform_policy)


