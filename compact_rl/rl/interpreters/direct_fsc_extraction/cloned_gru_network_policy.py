import tf_agents
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from keras import optimizers

from compact_rl.rl.tools.evaluation_results_class import EvaluationResults
from compact_rl.rl.tools.args_emulator import ArgsEmulator
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.tools.evaluators import evaluate_policy_in_model
from compact_rl.rl.interpreters.direct_fsc_extraction.extraction_stats import ExtractionStats

from compact_rl.rl.interpreters.direct_fsc_extraction.networks.gru_actor_network import GRUActorNetwork

DEBUG = True

import logging

import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClonedGRUActorPolicy(TFPolicy):
    def __init__(self, original_policy: TFPolicy,
                 observation_and_action_constraint_splitter=None,
                 model_name: str = "generic_model",
                 max_episode_length: int = 1000,
                 observation_length: int = 0,
                 lstm_units: int = 32,):
        self.original_policy = original_policy
        policy_state_spec = BoundedArraySpec(
            shape=(lstm_units,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='policy_state')
        super(ClonedGRUActorPolicy, self).__init__(time_step_spec=original_policy.time_step_spec,
                                                   action_spec=original_policy.action_spec,
                                                   policy_state_spec=policy_state_spec,
                                                   observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self.lstm_units = lstm_units
        self.lstm_actor : GRUActorNetwork = GRUActorNetwork(
            original_policy.time_step_spec.observation["observation"].shape[-1],
            original_policy.action_spec.maximum - original_policy.action_spec.minimum + 1,
            lstm_units=self.lstm_units
        )
        self.model_name = model_name
        self.max_episode_length = max_episode_length
        self.observation_length = observation_length


    def _variables(self):
        return self.lstm_actor.variables

    def distro(self, time_step, policy_state, seed):
        observation, mask = self.observation_and_action_constraint_splitter(
            time_step.observation)
        observation = tf.reshape(observation, (observation.shape[0], 1, -1))
        step_type = tf.reshape(time_step.step_type,
                               (time_step.step_type.shape[0], 1, -1))
        step_type = tf.cast(step_type, tf.float32)
        action, memory = self.lstm_actor(
            observation, step_type, policy_state, training=False)
        action = tf.reshape(action, (action.shape[0], -1))
        # Change logits of illegal actions to -inf
        action = tf.where(mask, action, -1e20)
        # memory = memory[:, -1, :]
        return action, memory

    def _action(self, time_step, policy_state, seed):
        action_probs, memory = self.distro(
            time_step, policy_state, seed)
        action = tf.random.categorical(action_probs, 1, dtype=tf.int32)
        # sample the most probable action
        # action = tf.argmax(action_probs, axis=-1, output_type=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        policy_step = PolicyStep(action=action, state=memory)
        # print(policy_step)
        return policy_step

    def _get_initial_state(self, batch_size: int):
        init_state = self.lstm_actor.get_initial_state(batch_size)
        return tf.reshape(init_state, (batch_size, self.lstm_units))

    def behavioral_clone_original_policy_to_fsc(self, buffer: TFUniformReplayBuffer, num_epochs: int,
                                                sample_len=32,
                                                environment: EnvironmentWrapperVec = None,
                                                tf_environment: TFPyEnvironment = None,
                                                args: ArgsEmulator = None,
                                                extraction_stats: ExtractionStats = None,
                                                learn_probs_regression=False) -> ExtractionStats:
        cloned_actor = self
        dataset_options = tf.data.Options()
        dataset_options.experimental_deterministic = False
        dataset = (
            buffer.as_dataset(sample_batch_size=64, num_steps=sample_len,
                              num_parallel_calls=tf.data.AUTOTUNE)
            .with_options(dataset_options)
            .prefetch(tf.data.AUTOTUNE)
        )

        if extraction_stats is None:
            extraction_stats = ExtractionStats(
                original_policy_reachability=0,
                original_policy_reward=0,
                number_of_samples=sample_len,
            )

        iterator = iter(dataset)
        neural_fsc = cloned_actor.lstm_actor
        optimizer = optimizers.Adam(learning_rate=1.6e-4, weight_decay=1e-5)

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
        def train_step(experience):
            observations = experience.observation["observation"]
            if learn_probs_regression:
                logits = experience.policy_info["dist_params"]["logits"]
                gt_actions = tf.nn.softmax(logits, axis=-1)
                gt_actions = tf.reshape(gt_actions, (gt_actions.shape[0], -1, gt_actions.shape[-1]))
            else:
                gt_actions = experience.action
            step_types = tf.cast(experience.step_type, tf.float32)
            step_types = tf.reshape(step_types, (step_types.shape[0], -1, 1))
            with tf.GradientTape() as tape:
                played_action, mem = neural_fsc(
                    observations, step_type=step_types)
                if learn_probs_regression:
                    played_action = keras.activations.softmax(played_action, axis=-1)
                
                loss = loss_fn(gt_actions, played_action)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, neural_fsc.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(
                zip(grads, neural_fsc.trainable_variables))
            loss_metric.update_state(loss)
            accuracy_metric.update_state(gt_actions, played_action)
            return loss

        for i in range(num_epochs):
            try:
                experience, _ = next(iterator)
            except StopIteration:  # Reset iteratoru, pokud dojde dataset
                iterator = iter(dataset)
                experience, _ = next(iterator)

            loss = train_step(experience) # train_step(experience)
            # Use the learning rate scheduler

            # Apply the LR scheduler to Adam optimizer 
            # Use the learning rate from the scheduler
            
            # print(neural_fsc.memory_dense.weights)
            self.periodical_evaluation(i, loss_metric, accuracy_metric, cloned_actor,
                                       environment, tf_environment, extraction_stats,
                                       self.evaluation_result)

        return extraction_stats

    def periodical_evaluation(self,
                              iteration_number: int,
                              loss_metric: keras.metrics.Mean,
                              accuracy_metric: keras.metrics.SparseCategoricalAccuracy,
                              cloned_actor: TFPolicy,
                              environment: EnvironmentWrapperVec,
                              tf_environment: TFPyEnvironment,
                              extraction_stats: ExtractionStats,
                              evaluation_result: EvaluationResults):
        if iteration_number % 100 == 0:
            avg_loss = loss_metric.result()
            accuracy = accuracy_metric.result()
            logger.info(f"Epoch {iteration_number}, Loss: {avg_loss:.4f}")
            logger.info(f"Epoch {iteration_number}, Accuracy: {accuracy:.4f}")
            extraction_stats.add_evaluation_accuracy(accuracy)
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            
        if iteration_number % 5000 == 0:
            self.evaluation_result = evaluate_policy_in_model(
                cloned_actor, None, environment, tf_environment, self.max_episode_length * 2, evaluation_result)
            extraction_stats.add_extraction_result(
                self.evaluation_result.reach_probs[-1], self.evaluation_result.returns[-1])
