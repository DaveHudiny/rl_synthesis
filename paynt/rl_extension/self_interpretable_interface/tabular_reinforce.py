import numpy as np
import tensorflow as tf
from compact_rl.rl.tools.args_emulator import ArgsEmulator

from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment

from compact_rl.rl.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy, initialize_random_independent_fsc_function

from compact_rl.rl.tools.evaluators import evaluate_policy_in_model
from compact_rl.rl.tests.general_test_tools import init_args, init_environment

import tqdm

class TabularReinforce:
    """A simple implementation of the REINFORCE algorithm for training a factored tabular FSC.
    """
    @staticmethod
    def collect_trajectories(policy : TableBasedPolicy, tf_env : TFPyEnvironment):
        env_state = tf_env.reset()
        policy_state = policy.get_initial_state(tf_env.batch_size)
        action_function = tf.function(policy.action)
        
        states, actions, rewards, is_dones = [], [], [], []

        for t in range(401):
            policy_step = action_function(env_state, policy_state)
            action = policy_step.action
            states.append((env_state.observation["integer"] * policy_state).numpy().reshape(-1)) # Combine observation and memory state to a single number
            is_dones.append(env_state.step_type.numpy().T)
            policy_state = policy_step.state
            
            
            env_state = tf_env.step(action)
            actions.append(action.numpy().T)
            rewards.append(env_state.reward.numpy().T)
            
        states = np.array(states).transpose()  # Shape: (batch_size, time_steps)
        actions = np.array(actions).transpose()  # Shape: (batch_size, time_steps)
        rewards = np.array(rewards).transpose()  # Shape: (batch_size, time_steps)
        is_dones = np.array(is_dones).transpose()
        return states, actions, rewards, is_dones

    @staticmethod
    def split_to_episodes(states, actions, rewards, is_dones):
        batch_size = len(states)
        episodes_states = []
        episodes_actions = []
        episodes_rewards = []
        for b in range(batch_size):
            episode_states = []
            episode_actions = []
            episode_rewards = []
            for t in range(states.shape[1]):
                if is_dones[b, t] == 0:
                    episode_states = [states[b, t]]
                    episode_actions = [actions[b, t]]
                    episode_rewards = [rewards[b, t]]
                elif is_dones[b, t] == 1:
                    episode_states.append(states[b, t])
                    episode_actions.append(actions[b, t])
                    episode_rewards.append(rewards[b, t])
                elif is_dones[b, t] == 2:
                    episodes_states.append(episode_states)
                    episodes_actions.append(episode_actions)
                    episodes_rewards.append(episode_rewards)

        # compute number of 2 in is_dones
        num_episodes = np.sum(is_dones == 2)

        return episodes_states, episodes_actions, episodes_rewards
    


    @staticmethod
    def compute_returns(rewards_episode, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards_episode):
            R = r + gamma * R
            returns.insert(0, R)
        # Odečtení průměrného returnu jako baseline
        returns = np.array(returns)
        returns -= np.mean(returns)  # nebo použij odhad hodnoty stavu (value function)
        return returns
    
    @staticmethod
    def update_policy(policy: TableBasedPolicy, states, actions, returns, learning_rate=0.01):
        action_table = policy.tf_observation_to_action_table.numpy()
        logits_table = np.log(action_table + 1e-8)  # Převod pravděpodobností na logity pro stabilitu
        for state, action, G in zip(states, actions, returns):
            memory = state % action_table.shape[0]
            observation = state // action_table.shape[0]
            # Gradient log-pravděpodobnosti: 1 / π(a|s)
            grad = 1 / (action_table[memory, observation, action] + 1e-8)
            # Aktualizace "logitů" (nebo přímo pravděpodobností) s entropy bonusem
            logits_table[memory, observation, action] += np.clip(grad * G + 0.01 * (1 - action_table[memory, observation, action]), -0.5, 0.5) * learning_rate
            # Softmax normalizace (pro celý řádek)

        exp_logits = np.exp(logits_table - np.max(logits_table, axis=2, keepdims=True))
        action_table = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)
        
        policy.tf_observation_to_action_table = tf.Variable(action_table, dtype=tf.float32)

    @staticmethod
    def train(fsc: TableBasedPolicy, environment: EnvironmentWrapperVec, args: ArgsEmulator = None):
        tf_env = TFPyEnvironment(environment)
        num_states = fsc.tf_observation_to_action_table.shape[0] * fsc.tf_observation_to_action_table.shape[1]
        num_actions = len(environment.action_keywords)
        initial_lr = 0.01
        decay_rate = 0.1  # Rychlost snižování learning rate
        tqdm_counter = tqdm.tqdm(range(10))
        print(f"Training on {num_states} states and {num_actions} actions")
        all_states, all_actions, all_returns = [], [], []
        for episode in tqdm_counter:
            # Adaptivní learning rate
            learning_rate = initial_lr / (1 + decay_rate * episode)

            # Sbírání trajektorií
            states, actions, rewards, is_dones = TabularReinforce.collect_trajectories(fsc, tf_env)
            episodes_states, episodes_actions, episodes_rewards = TabularReinforce.split_to_episodes(states, actions, rewards, is_dones)
            old_table = fsc.tf_observation_to_action_table.numpy().copy()
            # Aktualizace politiky pro každou epizodu
            for episode_states, episode_actions, episode_rewards in zip(episodes_states, episodes_actions, episodes_rewards):
                returns = TabularReinforce.compute_returns(episode_rewards)
                all_states.extend(episode_states)
                all_actions.extend(episode_actions)
                all_returns.extend(returns)
            TabularReinforce.update_policy(fsc, all_states, all_actions, all_returns, learning_rate)
            # Kontrola změny tabulky
            table_change = np.sum(np.abs(fsc.tf_observation_to_action_table.numpy() - old_table))
            tqdm_counter.set_description(f"Episode {episode+1}, Table change: {table_change:.4f}")
            # Evaluace
            if episode % 10 == 0:
                evaluate_policy_in_model(fsc, args, environment, tf_env)

if __name__ == "__main__":
    args = init_args("rl_src/models/evade/sketch.templ", "rl_src/models/evade/sketch.props")

    env, tf_env = init_environment(args)
    action_function, update_function = initialize_random_independent_fsc_function(env.action_keywords, env.stormpy_model.nr_observations, num_fsc_states=3)
    fsc = TableBasedPolicy(None, action_function, update_function, 0, env.action_keywords, nr_observations=env.stormpy_model.nr_observations, time_step_spec=env.time_step_spec(), action_spec=env.action_spec())

    TabularReinforce.train(fsc, env)
