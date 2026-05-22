from tf_agents.networks import network
import tensorflow as tf

import tf_agents
from tf_agents.environments import tf_py_environment

from stormpy.storage import SparsePomdp

import numpy as np

from enum import Enum


def create_recurrent_value_net_demasked_tuned(tf_environment: tf_py_environment.TFPyEnvironment):
    # preprocessing_layer = tf.keras.layers.Dense(32, activation='relu')
    input_layer_params = (32,)
    output_layer_params = (32,)
    value_net = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
        tf_environment.observation_spec()["observation"],
        # preprocessing_layers=preprocessing_layer,
        input_fc_layer_params=input_layer_params,
        output_fc_layer_params=output_layer_params,
        lstm_size=(32,),
        conv_layer_params=None
    )
    return value_net


def create_recurrent_value_net_demasked(tf_environment: tf_py_environment.TFPyEnvironment, rnn_less=False, width_of_lstm=32):
    if rnn_less:
        lstm_size = None
        value_net = tf_agents.networks.value_network.ValueNetwork(
            tf_environment.observation_spec()["observation"],
            fc_layer_params=(64, 64, 64)
            # preprocessing_layers=preprocessing_layers
        )
    else:
        lstm_size = (width_of_lstm,)
        value_net = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
            tf_environment.observation_spec()["observation"],
            # preprocessing_layers=preprocessing_layer,
            input_fc_layer_params=(64, ),
            lstm_size=lstm_size,
            output_fc_layer_params=(64,),
            conv_layer_params=None
        )
    return value_net