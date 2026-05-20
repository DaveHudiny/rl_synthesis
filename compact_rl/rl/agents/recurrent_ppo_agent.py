
import logging
from compact_rl.rl.agents.father_agent import FatherAgent
from compact_rl.rl.shielding.shielding_options import ShieldingOptions
from compact_rl.rl.tools.encoding_methods import *

import tensorflow as tf

from compact_rl.rl.environment import tf_py_environment
# from tf_agents.agents.ppo import ppo_agent
from compact_rl.rl.agents.tf_agents_modif import ppo_agent


from compact_rl.rl.environment.environment_wrapper import Environment_Wrapper

from compact_rl.rl.agents.policies.policy_mask_wrapper import PolicyMaskWrapper

from compact_rl.rl.agents.networks.value_networks import create_recurrent_value_net_demasked
from compact_rl.rl.agents.networks.actor_networks import create_recurrent_actor_net_demasked
from compact_rl.rl.agents.networks.fsc_like_network import FSCLikeNetwork

from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from compact_rl.rl.agents.tf_agents_modif.actor_distribution_rnn_network import ActorDistributionRnnNetwork

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from compact_rl.rl.agents.alternative_training.active_pretraining import EntropyRewardGenerator

from compact_rl.rl.tools.args_emulator import ArgsEmulator, ReplayBufferOptions
from paynt.quotient.fsc import FscFactored

from keras.optimizers import Adam
from compact_rl.rl.shielding.shield_processor import ShieldProcessor
from compact_rl.rl.shielding.shielded_dynamic_step_driver import ShieldedDynamicStepDriver

import sys
sys.path.append("../")


logger = logging.getLogger(__name__)


class Recurrent_PPO_agent(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args: ArgsEmulator, load=False, agent_folder=None, actor_net: ActorDistributionRnnNetwork = None,
                 critic_net: ValueRnnNetwork = None, discrete_actor_memory: bool = False):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        train_step_counter = tf.Variable(0)
        optimizer = Adam(
            learning_rate=args.learning_rate, beta_1=0.99, beta_2=0.99, weight_decay=0.0001)
        tf_environment = self.tf_environment
        action_spec = tf_environment.action_spec()
        if actor_net is not None:
            self.actor_net = actor_net
        elif discrete_actor_memory:
            self.actor_net = FSCLikeNetwork(
                tf_environment.observation_spec()["observation"],
                action_spec,
                memory_length=2,
                lstm_size=(32,),
                input_fc_layer_params=(64,),
            )
        else:
            self.actor_net = create_recurrent_actor_net_demasked(
                tf_environment, action_spec, rnn_less=self.args.use_rnn_less, width_of_lstm=self.args.width_of_lstm)
        if critic_net is not None:
            self.value_net = critic_net
        else:
            self.value_net = create_recurrent_value_net_demasked(
                tf_environment, rnn_less=self.args.use_rnn_less, width_of_lstm=self.args.width_of_lstm)

        time_step_spec = tf_environment.time_step_spec()
        time_step_spec = time_step_spec._replace(
            observation=tf_environment.observation_spec()["observation"])
        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            optimizer,
            actor_net=self.actor_net,
            value_net=self.value_net,
            num_epochs=3,
            train_step_counter=train_step_counter,
            greedy_eval=self.args.completely_greedy,
            discount_factor=self.args.discount_factor,
            use_gae=True,
            lambda_value=0.95,
            # policy_l2_reg=0.001,
            # value_function_l2_reg=0.001,
            entropy_regularization=0.02,
            normalize_rewards=True,
            normalize_observations=True,
            importance_ratio_clipping=0.2,
        )
        self.agent.initialize()
        print("Agent initialized with actor net:",
              self.agent.actor_net.summary())
        print("Agent initialized with value net:",
              self.agent._value_net.summary())
        logging.info("Agent initialized")
        self.init_replay_buffer()
        logging.info("Replay buffer initialized")

        self.init_collector_driver(self.tf_environment, demasked=True)
        if self.args.predicate_automata_obs or self.args.curiosity_automata_reward or self.args.go_explore:
            predicate_automata = self.environment.predicate_automata
        else:
            predicate_automata = None

        self.wrapper = PolicyMaskWrapper(self.agent.policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec(),
                                         is_greedy=(not self.args.prefer_stochastic), predicate_automata=predicate_automata)
        # self.wrapper.set_policy_masker()
        self.wrapper_eager = PyTFEagerPolicy(self.wrapper, True, False)
        logging.info("Collector driver initialized")
        if self.args.entropy_reward:
            entropy_reward_generator = EntropyRewardGenerator(binary_flag=args.use_binary_entropy_reward,
                                                              full_observability_flag=args.full_observable_entropy_reward,
                                                              max_reward=1.0, decreaser='halve')
            self.init_pretraining_driver(entropy_reward_generator)
        if load:
            self.load_agent()
        # self.init_vec_evaluation_driver(
        #     self.tf_environment, self.environment, num_steps=self.args.max_steps)
        # logging.info("Evaluation driver initialized")

    def reinit_safe_collector(self, shield_processor: ShieldProcessor = None, 
                              shielded_option: ShieldingOptions = ShieldingOptions.NAIVE_TRAINING_SHIELDING):
        if shielded_option is not ShieldingOptions.NO_SHIELDING:
            assert shield_processor, "Shield processor must be provided for safe collector reinitialization."
        observers = [self.get_demasked_observer(
            self.args.vectorized_envs_flag)]
        eager = PyTFEagerPolicy(self.collect_policy_wrapper, True, False)
        self.driver = ShieldedDynamicStepDriver(
            env=self.tf_environment,
            policy=eager,
            observers=observers,
            num_steps=self.num_steps,
            shield_processor=shield_processor,
            shielded_training=(shielded_option != ShieldingOptions.EVALUATION_ONLY_SHIELDING)
        )

    def train_agent_shielded(self, iterations: int, vectorized: bool = True,
                             replay_buffer_option: ReplayBufferOptions = ReplayBufferOptions.ON_POLICY,
                             fsc: FscFactored = None,
                             jumpstart_fsc: bool = False,
                             debug: bool = False,
                             shaping=False,
                             shield_processor: ShieldProcessor = None,
                             shielded_option: ShieldingOptions = ShieldingOptions.NAIVE_TRAINING_SHIELDING):
        self.reinit_safe_collector(shield_processor, shielded_option)

        self.train_agent(iterations, vectorized, replay_buffer_option,
                         fsc, jumpstart_fsc, debug, shaping)

    def special_agent_pretraining_stuff(self):
        logger.info("Setting value net to trainable")
        for var in self.agent.trainable_variables:
            var._trainable = False
        for var in self.agent._value_net.variables:
            var._trainable = True

    def special_agent_midtraining_stuff(self):
        logger.info("Setting actor net to trainable")
        for var in self.agent.variables:
            var._trainable = True
        self.agent.initialize()

    def set_policy_masking(self):
        """If PPO, this function sets the masking active for agent wrapper."""
        print("Masker set")
        self.wrapper.set_policy_masker()

    def unset_policy_masking(self):
        """If PPO, this function sets the masking inactive for agent wrapper."""
        print("Masker unset")
        self.wrapper.set_identity_masker()

    def set_agent_greedy(self):
        """Set the agent for to be greedy for evaluation. Used only with PPO agent, where we select greedy evaluation.
        """
        print("Setting agent to greedy")
        self.wrapper.set_greedy(True)

    def set_agent_stochastic(self):
        print("Setting agent to stochastic")
        self.wrapper.set_greedy(False)

    def reset_weights(self, value_only: bool = False):

        for var in self.agent.variables:
            if value_only:
                if "Value" not in var.name:
                    continue
            if "kernel" in var.name:
                if "dynamic_unroll" in var.name:  # Rekurentní vrstvy
                    # Použití Glorot Uniform pro RNN váhy
                    glorot_stddev = tf.sqrt(
                        2.0 / (var.shape[0] + var.shape[1]))
                    var.assign(tf.random.uniform(
                        var.shape, minval=-glorot_stddev, maxval=glorot_stddev))
                else:  # Dense vrstvy
                    # Použití Variance Scaling s truncated normal pro Dense váhy
                    scale = 2.0
                    stddev = tf.sqrt(scale / var.shape[0])
                    var.assign(tf.random.truncated_normal(
                        var.shape, stddev=stddev))
            elif "bias" in var.name:
                # Inicializace biasů na nulu
                var.assign(tf.zeros(var.shape))
        self.agent.initialize()

    def shrink_and_perturb(self):
        """Implements the shrink and perturb method for the agent's actor and critic networks.
        """
        logger.info("Shrinking and perturbing actor and critic networks")
        for var in self.agent.variables:
            if "kernel" in var.name:
                if "dynamic_unroll" in var.name:
                    # For RNN layers, we shrink the weights by a factor of 0.9 and add smaller Gaussian noise
                    weights = var.numpy()
                    # Shrink the weights by a factor of 0.9
                    weights *= 0.9
                    # Add Gaussian noise with mean 0 and standard deviation 0.0075
                    noise = np.random.normal(0, 0.075, weights.shape)
                    weights += noise
                    var.assign(weights)
                else:
                    weights = var.numpy()
                    # Shrink the weights by a factor of 0.9
                    weights *= 0.9
                    # Add Gaussian noise with mean 0 and standard deviation 0.01
                    noise = np.random.normal(0, 0.1, weights.shape)
                    weights += noise
                    var.assign(weights)
            elif "bias" in var.name:
                # For biases, we shrink them by a factor of 0.9 and add smaller Gaussian noise
                biases = var.numpy()
                biases *= 0.9
                # Add Gaussian noise with mean 0 and standard deviation 0.001
                noise = np.random.normal(0, 0.1, biases.shape)
                biases += noise
                var.assign(biases)

        # Reinitialize the agent after perturbation
        self.agent.initialize()
        # Log the shrink and perturb action
        logger.info("Actor and critic networks shrunk and perturbed")
