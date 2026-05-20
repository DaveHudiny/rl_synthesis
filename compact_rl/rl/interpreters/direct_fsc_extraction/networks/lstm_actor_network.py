import tensorflow as tf
from tf_agents.trajectories.time_step import StepType

from keras import layers, models, activations
from tf_agents.keras_layers import dynamic_unroll_layer

class LSTMActorNetwork(models.Model):
    def __init__(self, observation_shape: tf.TensorShape,
                 action_range: int,
                 lstm_units: int = 32):
        super(LSTMActorNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_range = action_range
        self.dense1 = layers.Dense(128, activation='relu')
        self.lstm_layer = layers.GRU(
            lstm_units, return_sequences=True, return_state=True)
        self.lstm_units = lstm_units
        self.pre_action_dense = layers.Dense(64, activation='relu')
        self.pre_action_dense_2 = layers.Dense(32, activation='relu')
        self.action = layers.Dense(self.action_range, activation=None)

    def get_initial_state(self, batch_size):
        inputs = tf.zeros((batch_size, 1, self.observation_shape))
        return self.lstm_layer.get_initial_state(inputs)

    def call(self, inputs, step_type=None, old_memory=None, training=False):
        x = self.dense1(inputs)
        if old_memory is None:
            old_memory = self.lstm_layer.get_initial_state(inputs)
        
        x, final_memory_state = self.lstm_layer(x, initial_state=old_memory, training=training)
        x = self.pre_action_dense(x)
        x = self.pre_action_dense_2(x)
        action = self.action(x)
        return action, (final_memory_state)
