import stormpy
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

        row_sums = policy.sum(axis=-1)
        assert np.allclose(row_sums, 1.0), "Policy rows do not sum to 1."

        
        return policy

    def make_permissive_policy_uniform(self, policy):
        """
        Makes the permissive policy uniform by setting all non-zero probabilities to 1.0 and normalizing the policy.
        Args:
            policy (np.ndarray): A 2D array representing the permissive policy, where each row corresponds to an observation and each column corresponds to an action.
        Returns:
            uniform_policy (np.ndarray): A 2D array representing the uniform permissive policy.
        """
        uniform_policy = np.where(policy > 0.0, 1.0, 0.0)
        uniform_policy = uniform_policy / uniform_policy.sum(axis=-1, keepdims=True)
        return uniform_policy

    
    def compute_enabled_choices(self, nr_available_actions, state, observation, permissive_policy):
        enabled_choices = []
        for action_offset in range(nr_available_actions):
            
            choice_index = self.model.get_choice_index(state, action_offset)
            choice_labels = self.model.choice_labeling.get_labels_of_choice(choice_index)
            if len(choice_labels) == 0:
                policy_action_index = 0
            else:
                action_label = next(iter(choice_labels))
                if action_label not in self.environment.action_indices:
                    raise ValueError(
                        f"Action label '{action_label}' from the model was not found in environment action indices."
                    )
                policy_action_index = self.environment.action_indices[action_label]
            
            

            action_probability = float(permissive_policy[observation, policy_action_index])
            if action_probability > 0.0:
                
                enabled_choices.append((choice_index, action_probability))

        enabled_prob_sum = sum(probability for _, probability in enabled_choices)
        if enabled_prob_sum <= 0.0:
            raise ValueError(
                f"No enabled choice with positive probability for state {state} and observation {observation}."
            )
        return enabled_choices, enabled_prob_sum

    @staticmethod
    def init_reward_models(model : stormpy.storage.SparsePomdp):
        nr_states = model.nr_states
        state_action_rewards = {}
        state_rewards = {}
        for reward_name, reward_model in model.reward_models.items():
            if reward_model.has_state_action_rewards:
                state_action_rewards[reward_name] = np.zeros(nr_states, dtype=np.float64)
            elif reward_model.has_state_rewards:
                state_rewards[reward_name] = list(reward_model.state_rewards)
            else:
                raise NotImplementedError(
                    f"Reward model '{reward_name}' is not supported. "
                    "Only state-action and state rewards are currently supported."
                )
        return state_action_rewards, state_rewards

    @staticmethod
    def build_dtmc_components(model, dtmc_transition_matrix, state_action_rewards, state_rewards):
        dtmc_labeling = stormpy.storage.StateLabeling(model.nr_states)
        for label in model.labeling.get_labels():
            dtmc_labeling.add_label(label)
        for state in range(model.nr_states):
            for label in model.labeling.get_labels_of_state(state):
                dtmc_labeling.add_label_to_state(label, state)

        dtmc_reward_models = {}
        for reward_name, reward_values in state_action_rewards.items():
            dtmc_reward_models[reward_name] = stormpy.SparseRewardModel(
                optional_state_action_reward_vector=list(reward_values)
            )
        for reward_name, reward_values in state_rewards.items():
            dtmc_reward_models[reward_name] = stormpy.SparseRewardModel(
                optional_state_reward_vector=reward_values
            )

        components = stormpy.storage.SparseModelComponents(
            transition_matrix=dtmc_transition_matrix,
            state_labeling=dtmc_labeling,
            reward_models=dtmc_reward_models
        )
        return components
    

    def construct_permissive_dtmc(self, permissive_policy):
        """
        Constructs a permissive DTMC from the given permissive policy.
        Args:
            permissive_policy (np.ndarray): A 2D array representing the permissive policy, where each row corresponds to an observation and each column corresponds to an action.
        Returns:
            dtmc (stormpy.DTMC): A DTMC constructed from the permissive policy.
        """
        if permissive_policy.shape != (self.model.nr_observations, self.environment.nr_actions):
            raise ValueError(
                f"Expected permissive policy shape {(self.model.nr_observations, self.environment.nr_actions)}, "
                f"got {permissive_policy.shape}."
            )

        nr_states = self.model.nr_states
        dtmc_tm_builder = stormpy.SparseMatrixBuilder(
            rows=nr_states,
            columns=nr_states,
            force_dimensions=True
        )
        state_action_rewards, state_rewards = Permissive_DTMC_Extractor.init_reward_models(self.model)
        

        for state in range(nr_states):
            observation = self.model.observations[state]
            nr_available_actions = self.model.get_nr_available_actions(state)

            # Compute enabled choices and their probabilities based on the permissive policy
            enabled_choices, enabled_prob_sum = self.compute_enabled_choices(nr_available_actions, state, observation, permissive_policy)


            # Compute next state probabilities and expected rewards
            next_state_probabilities = {}
            expected_rewards = {name: 0.0 for name in state_action_rewards.keys()}

            for choice_index, raw_probability in enabled_choices:
                action_probability = raw_probability / enabled_prob_sum

                for reward_name, reward_model in self.model.reward_models.items():
                    if reward_model.has_state_action_rewards:
                        expected_rewards[reward_name] += (
                            float(reward_model.state_action_rewards[choice_index]) * action_probability
                        )

                for transition_entry in self.model.transition_matrix.get_row(choice_index):
                    next_state = transition_entry.column
                    next_state_probabilities[next_state] = (
                        next_state_probabilities.get(next_state, 0.0)
                        + float(transition_entry.value()) * action_probability
                    )

            for reward_name, reward_value in expected_rewards.items():
                state_action_rewards[reward_name][state] = reward_value

            for next_state, probability in next_state_probabilities.items():
                dtmc_tm_builder.add_next_value(state, next_state, probability)


        dtmc_transition_matrix = dtmc_tm_builder.build()

        components = Permissive_DTMC_Extractor.build_dtmc_components(self.model, dtmc_transition_matrix, state_action_rewards, state_rewards)
        return stormpy.storage.SparseDtmc(components)

    def extract_dtmc(self, num_steps=1000, nr_environments=256, threshold=0.1) -> stormpy.storage.SparseDtmc:
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
        dtmc = self.construct_permissive_dtmc(uniform_policy)
        return dtmc
