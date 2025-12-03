from rl_src.agents.tf_agents_modif import actor_distribution_rnn_network
from tf_agents.networks import actor_distribution_network
from rl_src.environment import tf_py_environment

import tensorflow as tf
import numpy as np


def create_recurrent_actor_net_demasked_tuned(tf_environment: tf_py_environment.TFPyEnvironment, action_spec):
    preprocessing_layer = tf.keras.layers.Dense(32, activation='relu')
    # input_layer_params = (64,)
    output_layer_params = (32,)
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        tf_environment.observation_spec()["observation"],
        action_spec,
        preprocessing_layers=preprocessing_layer,
        # input_fc_layer_params=input_layer_params,
        output_fc_layer_params=output_layer_params,
        lstm_size=(32,),
        conv_layer_params=None
    )
    return actor_net


def create_recurrent_actor_net_demasked(tf_environment: tf_py_environment.TFPyEnvironment, action_spec, rnn_less=False, width_of_lstm=32):
    if rnn_less:
        lstm_size = None
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            tf_environment.observation_spec()["observation"],
            action_spec,
            fc_layer_params=(64, 64, 64)
            # preprocessing_layers=preprocessing_layers,
            # conv_layer_params=None
        )
    else:
        lstm_size = (width_of_lstm,)
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            tf_environment.observation_spec()["observation"],
            action_spec,
            # preprocessing_layers=preprocessing_layer,
            input_fc_layer_params=(64,),
            # output_fc_layer_params=(64,),
            output_fc_layer_params=(64,),
            lstm_size=lstm_size,
            conv_layer_params=None
        )
    return actor_net
