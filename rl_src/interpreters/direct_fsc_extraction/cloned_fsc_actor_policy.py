import tf_agents
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import keras
from keras import backend as K

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from keras import optimizers

from rl_src.interpreters.direct_fsc_extraction.networks.fsc_like_actor_network import FSCLikeActorNetwork
from rl_src.interpreters.direct_fsc_extraction.networks.fsc_like_dict_actor_network import FSCLikeDictActorNetwork
from rl_src.tools.evaluation_results_class import EvaluationResults
from rl_src.tools.specification_check import SpecificationChecker
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.tools.evaluators import evaluate_policy_in_model
from rl_src.interpreters.direct_fsc_extraction.extraction_stats import ExtractionStats

from rl_src.interpreters.direct_fsc_extraction .networks.lstm_actor_network import LSTMActorNetwork

DEBUG = True

import logging

import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClonedFSCActorPolicy(TFPolicy):
    def __init__(self, original_policy: TFPolicy, memory_size: int,
                 observation_and_action_constrint_splitter=None,
                 use_one_hot=True,
                 use_residual_connection=True,
                 optimization_specification: SpecificationChecker.Constants = SpecificationChecker.Constants.REACHABILITY,
                 model_name: str = "generic_model",
                 find_best_policy: bool = False,
                 max_episode_length: int = 1000,
                 observation_length: int = 0,
                 orig_env_use_stacked_observations: bool = True,
                 use_gumbel_softmax: bool = False,
                 seed=42,
                 fsrs_learning: bool = True):
        self.original_policy = original_policy
        self.use_one_hot = use_one_hot
        policy_state_spec = BoundedArraySpec(
            (memory_size,), np.float32, minimum=-1.5, maximum=1.5)
        super(ClonedFSCActorPolicy, self).__init__(time_step_spec=original_policy.time_step_spec,
                                                   action_spec=original_policy.action_spec,
                                                   policy_state_spec=policy_state_spec,
                                                   observation_and_action_constraint_splitter=observation_and_action_constrint_splitter)
        self.memory_size = memory_size
        self.fsrs_learning = fsrs_learning
        self.fsc_actor = FSCLikeActorNetwork(
            observation_length,
            original_policy.action_spec.maximum + 1,
            memory_size,
            use_one_hot=use_one_hot,
            gumbel_softmax_one_hot=use_gumbel_softmax,
            fsrs_learning=fsrs_learning)
        self.model_name = model_name
        self.optimization_specification = optimization_specification
        self.find_best_policy = find_best_policy
        self.max_episode_length = max_episode_length
        self.observation_length = observation_length
        self.orig_env_use_stacked_observations = orig_env_use_stacked_observations
        self.use_gumbel_softmax = use_gumbel_softmax
        self.seed = tfp.util.SeedStream(seed, salt="cloned_fsc_actor_policy")


    def set_probs_updates(self):
        self.fsc_actor.set_return_probs(True)

    def unset_probs_updates(self):
        self.fsc_actor.set_return_probs(False)

    def load_best_policy(self):
        try:
            self.fsc_actor.load_weights(
                f"experiments_models/{self.model_name}/best_policy.h5")
        except Exception as e:
            logger.error(f"Could not load best policy: {e}")

    def save_best_policy(self):
        os.makedirs(os.path.dirname(f"experiments_models/{self.model_name}/best_policy.h5"), exist_ok=True)
        self.fsc_actor.save_weights(
            f"experiments_models/{self.model_name}/best_policy.h5")

    def _variables(self):
        return self.fsc_actor.variables

    def distro(self, time_step, policy_state, seed):
        observation, mask = self.observation_and_action_constraint_splitter(
            time_step.observation)
        observation = tf.reshape(observation, (observation.shape[0], 1, -1))
        if self.orig_env_use_stacked_observations:
            observation = observation[:, :, :self.observation_length]
        policy_state = tf.reshape(policy_state, (policy_state.shape[0], -1))
        step_type = tf.reshape(time_step.step_type,
                               (time_step.step_type.shape[0], 1, -1))
        step_type = tf.cast(step_type, tf.float32)
        action, memory = self.fsc_actor(
            observation, step_type, policy_state, training=False, seed=seed)
        action = tf.reshape(action, (action.shape[0], -1))
        # Change logits of illegal actions to -inf
        action = tf.where(mask, action, -1e20)
        # memory = memory[:, -1, :]
        return action, memory

    def _action(self, time_step, policy_state, seed):
        action_probs, memory = self.distro(
            time_step, policy_state, seed)
        action = tf.random.categorical(action_probs, 1, dtype=tf.int32, seed=seed)
        # sample the most probable action
        # action = tf.argmax(action_probs, axis=-1, output_type=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        policy_step = PolicyStep(action=action, state=memory)
        # print(policy_step)
        return policy_step

    def _get_initial_state(self, batch_size: int):
        init_state = self.fsc_actor.get_initial_state(batch_size)
        return tf.reshape(init_state, (batch_size, -1))
    
    def compute_inverted_global_action_distribution(self, buffer: TFUniformReplayBuffer, nr_actions, use_probs_regression=False):
        """ Computes the inverted global action distribution from the replay buffer, that is used to compute weights for the cross entropy weights.
        Args:
            buffer (TFUniformReplayBuffer): The replay buffer containing the experience.
            nr_actions (int): The number of actions in the environment.
        Returns:
            tf.Tensor: A tensor containing the inverted global action distribution."""
        data = buffer.gather_all()
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(64).prefetch(tf.data.AUTOTUNE)
        action_counts_or_probs = tf.zeros((nr_actions,), dtype=tf.float32)
        for experience in dataset:
            if use_probs_regression:
                logits = experience.policy_info["dist_params"]["logits"]
                logits = tf.reshape(logits, (-1, logits.shape[-1]))
                probs = tf.nn.softmax(logits, axis=-1)
                action_counts_or_probs += tf.reduce_sum(probs, axis=0)
            else:
                actions = experience.action
                actions = tf.reshape(actions, (-1,))
                action_counts += tf.math.bincount(actions, minlength=nr_actions)
        action_counts_or_probs = tf.cast(action_counts_or_probs, tf.float32)
        action_counts_or_probs = tf.where(
            action_counts_or_probs == 0, tf.ones_like(action_counts_or_probs), action_counts_or_probs)
        probs = tf.math.reciprocal(action_counts_or_probs)
        probs = probs / tf.reduce_sum(probs)
        inverted_weights = 1 / probs
        return inverted_weights
    
    def get_weighted_cross_entropy_loss(self, weights, nr_actions):
        """ Returns a weighted cross entropy loss function.
        Args:
            weights (tf.Tensor): The weights of each class (used for the cross entropy loss).
            nr_actions (int): The number of actions in the environment.
        Returns:
            function: A loss function that takes logits and labels as input and returns the weighted cross entropy loss.
        """
        weights = tf.constant(weights, dtype=tf.float32)
        def weighted_cross_entropy_loss(gt_probs, logits): # Data are in shape (batch_size, trajectory_length, nr_actions)
            element_wise = tf.math.multiply(gt_probs, logits)
            weighted_element_wise = tf.math.multiply(element_wise, weights)
            loss = -tf.reduce_sum(weighted_element_wise)
            return loss
        return weighted_cross_entropy_loss
    
    def schedule_gumbel_temperature(self, epoch: int, network: FSCLikeActorNetwork, total_epochs=10000):
        """Schedules the Gumbel temperature using cosine annealing.

        Args:
            epoch (int): The current epoch (0 to 19999).
            network (FSCLikeActorNetwork): The network to set the temperature for.
        """
        import math

        tau_max = 3.0
        tau_min = 0.1
        total_epochs = total_epochs

        # cosine annealing
        tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + math.cos(math.pi * epoch / total_epochs))

        network.set_gumbel_temperature(tau)
            

    def behavioral_clone_original_policy_to_fsc(self, buffer: TFUniformReplayBuffer, num_epochs: int,
                                                sample_len=32,
                                                specification_checker: SpecificationChecker = None,
                                                environment: EnvironmentWrapperVec = None,
                                                tf_environment: TFPyEnvironment = None,
                                                args: ArgsEmulator = None,
                                                extraction_stats: ExtractionStats = None,
                                                learn_probs_regression=False) -> ExtractionStats:
        cloned_actor = self
        dataset_options = tf.data.Options()
        dataset_options.deterministic = True

        dataset = (
            buffer.as_dataset(sample_batch_size=64, num_steps=sample_len,
                              num_parallel_calls=1)
            .with_options(dataset_options)
            .prefetch(64)
        )
        iterator = iter(dataset)

        if extraction_stats is None:
            extraction_stats = ExtractionStats(
                original_policy_reachability=0,
                original_policy_reward=0,
                use_one_hot=self.use_one_hot,
                number_of_samples=sample_len,
                memory_size=self.memory_size,
                residual_connection=self.fsc_actor.use_residual_connection
            )

        
        neural_fsc : FSCLikeActorNetwork = cloned_actor.fsc_actor
        neural_fsc.add_noise_to_neural_weights(self.seed())
        optimizer = optimizers.Adam(learning_rate=1.6e-4, weight_decay=1e-5)
        
        if self.fsrs_learning:
            loss_fn = keras.losses.MeanSquaredError()
            accuracy_metric = keras.metrics.MeanSquaredError(
                name="mse_accuracy")
        else:
            if learn_probs_regression:
                # loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
                # loss_fn = keras.losses.MeanSquaredError()
                loss_fn = keras.losses.KLDivergence()
                # loss_fn = self.get_weighted_cross_entropy_loss(
                #     weights_normalized, len(environment.action_keywords))
                accuracy_metric = keras.metrics.CategoricalAccuracy(
                    name="accuracy"
                )
            else:
                loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                accuracy_metric = keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy")
        loss_metric = keras.metrics.Mean(name="train_loss")

        self.evaluation_result = None
        observation_length = environment.observation_spec_len
        
        @tf.function
        def train_step_2(experience):
            observations = experience.observation["observation"]
            if environment.use_stacked_observations:
                observations = observations[:, :, :observation_length]

            if learn_probs_regression:
                logits = experience.policy_info["dist_params"]["logits"]
                gt_actions = tf.nn.softmax(logits, axis=-1)
                gt_actions = tf.reshape(gt_actions, (gt_actions.shape[0], -1, gt_actions.shape[-1]))
            else:
                gt_actions = experience.action

            step_types = tf.cast(experience.step_type, tf.float32)
            step_types = tf.reshape(step_types, (step_types.shape[0], -1, 1))

            batch_size, T, _ = observations.shape
            old_memory = None

            # Add small noise to observations
            # observations += tf.random.normal(shape=tf.shape(observations), mean=0.0, stddev=0.2, seed=self.seed())

            with tf.GradientTape() as tape:
                total_loss = 0.0
                for t in range(T):
                    if t == T - 1:
                        break
                    current_obs = observations[:, t, :]
                    if self.fsrs_learning:
                        # Append gt action to the observation for FSRS learning
                        act = gt_actions[:, t]
                        act = tf.reshape(act, (act.shape[0], 1))
                        act = tf.cast(act, tf.float32)
                        current_obs = tf.concat([current_obs, act], axis=-1)
                    next_obs = observations[:, (t + 1), :]
                    next_obs = tf.reshape(next_obs, (next_obs.shape[0], -1))
                    current_step_type = step_types[:, t, :]
                    current_obs = tf.reshape(current_obs, (current_obs.shape[0], 1, -1))
                    current_step_type = tf.reshape(current_step_type, (current_step_type.shape[0], 1, -1))
                    played_action, old_memory = neural_fsc(
                        current_obs,
                        step_type=current_step_type,
                        old_memory=old_memory,
                        seed=self.seed()
                    )

                    # Výpočet ztráty pro aktuální krok
                    
                    
                    if self.fsrs_learning:
                        accuracy_metric.update_state(next_obs, played_action)
                        current_loss = loss_fn(next_obs, played_action)
                    else:
                        if not learn_probs_regression:
                            played_action_soft = keras.activations.softmax(played_action, axis=-1)
                            accuracy_metric.update_state(gt_actions[:, t], played_action_soft)
                            current_loss = loss_fn(gt_actions[:, t], played_action)
                        else:
                            accuracy_metric.update_state(gt_actions[:, t, :], played_action)
                            current_loss = loss_fn(gt_actions[:, t, :], played_action)
                    
                    total_loss += current_loss

                total_loss /= T
            grads = tape.gradient(total_loss, neural_fsc.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, neural_fsc.trainable_variables))

            loss_metric.update_state(total_loss)
                

            return total_loss

        self.set_probs_updates()
        for i in range(num_epochs):
            try:
                experience, _ = next(iterator)
            except StopIteration:
                iterator = iter(dataset)
                experience, _ = next(iterator)
            loss = train_step_2(experience) # train_step(experience)

            if True:
                self.schedule_gumbel_temperature(i, neural_fsc, total_epochs=num_epochs)
            self.periodical_evaluation(i, loss_metric, accuracy_metric, cloned_actor,
                                       environment, tf_environment, extraction_stats,
                                       self.evaluation_result, specification_checker)

        return extraction_stats

    def periodical_evaluation(self,
                              iteration_number: int,
                              loss_metric: keras.metrics.Mean,
                              accuracy_metric: keras.metrics.SparseCategoricalAccuracy,
                              cloned_actor: TFPolicy,
                              environment: EnvironmentWrapperVec,
                              tf_environment: TFPyEnvironment,
                              extraction_stats: ExtractionStats,
                              evaluation_result: EvaluationResults,
                              specification_checker: SpecificationChecker):
        if iteration_number % 100 == 0:
            avg_loss = loss_metric.result()
            accuracy = accuracy_metric.result()
            logger.info(f"Epoch {iteration_number}, Loss: {avg_loss:.4f}")
            logger.info(f"Epoch {iteration_number}, Accuracy: {accuracy:.4f}")
            extraction_stats.add_evaluation_accuracy(accuracy)
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            
        if False and iteration_number % 500 == 0:
            self.evaluation_result = evaluate_policy_in_model(
                cloned_actor, None, environment, tf_environment, self.max_episode_length * 2, evaluation_result)
            extraction_stats.add_extraction_result(
                self.evaluation_result.reach_probs[-1], self.evaluation_result.returns[-1])
            self.check_and_save(self.evaluation_result)

    def check_and_save(self, evaluation_result: EvaluationResults):
        if self.find_best_policy:
            if self.optimization_specification == SpecificationChecker.Constants.REACHABILITY:
                if tf.math.equal(evaluation_result.reach_probs[-1], tf.reduce_max(evaluation_result.reach_probs)):
                    self.save_best_policy()
            elif self.optimization_specification == SpecificationChecker.Constants.REWARD:
                if tf.math.equal(evaluation_result.returns[-1], tf.reduce_max(evaluation_result.returns)):
                    self.save_best_policy()
            else:
                raise ValueError("Unknown optimization specification")
