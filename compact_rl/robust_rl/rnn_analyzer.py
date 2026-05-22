import sklearn

import tensorflow as tf
import numpy as np

from sklearn.cluster import DBSCAN

from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent

from compact_rl.rl.tools.args_emulator import ArgsEmulator

class RNNAnalyzer:
    """
    Class for analyzing the RNN memory of an agent.
    """
    def __init__(self, args):
        """
        Initialize the RNNAnalyzer with the given arguments.
        """
        self.args = args

    def _collect_rnn_feedbacks(self, agent : Recurrent_PPO_agent, tf_environment : TFPyEnvironment):
        """
        Collect RNN feedbacks from the agent on the given tf_environment.
        """
        policy = agent.get_policy(False, True)
        tf_environment.reset()
        time_step = tf_environment.current_time_step()
        policy_state = policy.get_initial_state(tf_environment.batch_size)
        feedbacks = []
        for _ in range(self.args.max_steps + 1):
            action_step = policy.action(time_step, policy_state)
            time_step = tf_environment.step(action_step.action)
            policy_state = action_step.state
            dictless_feedback = policy_state["actor_network_state"]
            feedbacks.append(tf.concat([dictless_feedback], axis=-1))

        return feedbacks
    
    def dbscan_analysis(self, feedbacks):
        """
        Perform DBSCAN analysis on the collected feedbacks.
        """
        # Convert feedbacks to a suitable format for clustering
        feedbacks = [feedback.numpy() for feedback in feedbacks]
        feedbacks = np.array(feedbacks)
        feedbacks = feedbacks.reshape(-1, feedbacks.shape[-1]) # Clusters should be based on the last dimension
        # Shuffle the feedbacks to ensure randomness
        np.random.shuffle(feedbacks)
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(feedbacks[0:100000])
        return clustering
        

    
    def analyze(self, agent: Recurrent_PPO_agent, tf_environment: TFPyEnvironment):
        """
        Analyze the agent's RNN memory provided by simulations on tf_environment.
        """
        feedbacks = self._collect_rnn_feedbacks(agent, tf_environment)
        # Perform DBSCAN analysis on the collected feedbacks
        clustering = self.dbscan_analysis(feedbacks)
        labels = clustering.labels_
        number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("Number of clusters found:", number_of_clusters)
        return number_of_clusters