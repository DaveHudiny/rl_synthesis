"""Actor/value networks for training a learned risk-budget function via PPO on
`RiskBudgetTrainingEnv` (see that module's docstring for the overall setup).

The action is `max_pairs` independent Gaussian logits (Gaussian-in-logit-space,
per the confirmed design), one per (action, next_state) slot. Slots are
padding-masked (`pair_mask`) and - critically - which (action, next_state) pair
occupies slot k is an arbitrary artifact of dict enumeration order
(`reachable_pair_probs`), not a fixed identity that means the same thing across
states the way e.g. "torque on joint 3" does in a standard continuous-control
action space. So unlike `_normal_projection_net`'s usual use (one Dense layer
mapping an encoding directly to all `max_pairs` outputs, i.e. a distinct
learned weight per output/slot index), the per-slot mean here is produced by a
SINGLE shared small MLP applied identically to every slot, driven by that
slot's own features (which pair it actually is) rather than its arbitrary
index. This still reuses `_normal_projection_net`'s exact conventions (the
softplus/bias-initializer trick for turning `init_action_stddev` into a
starting stddev, and the same MultivariateNormalDiag-with-loc/scale-named-spec
backward-compatible construction `NormalProjectionNetwork` itself uses) - just
applied to a per-slot-shared computation instead of its single joint Dense.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import lstm_encoding_network
from tf_agents.networks import network
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec


def _shared_global_encoder(name, input_fc_layer_params, lstm_size, output_fc_layer_params,
                            global_input_dim):
    return lstm_encoding_network.LSTMEncodingNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([global_input_dim], tf.float32),
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        name=name + "_lstm_encoder")


def _global_input(observation):
    return tf.concat([observation["state_features"], observation["action_distribution"]], axis=-1)


def _mvn_diag_output_spec(max_pairs, name):
    # Adapted from NormalProjectionNetwork._output_distribution_spec's
    # is_multivariate branch: the input spec is named with Normal's own
    # parameter names ("loc"/"scale") for backward compatibility, even though
    # the distribution actually built is MultivariateNormalDiag (whose
    # constructor takes "scale_diag") - this is what makes
    # ppo_utils.get_distribution_params/distribution_from_spec's legacy
    # DistributionNetwork round-trip (used to reconstruct the old policy's
    # distribution from replay-buffer-logged params, for the PPO ratio) work;
    # an Independent(Normal(...), 1) would NOT round-trip since Independent's
    # own top-level parameters aren't tensors (its "distribution" parameter is
    # a nested Distribution object), so get_distribution_params would extract
    # nothing.
    sample_spec = tensor_spec.TensorSpec([max_pairs], tf.float32, "risk_budget_logits")
    param_properties = tfp.distributions.Normal.parameter_properties()
    input_param_spec = {
        key: tensor_spec.TensorSpec(shape=sample_spec.shape, dtype=sample_spec.dtype, name=f"{name}_{key}")
        for key in param_properties
    }

    def distribution_builder(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["scale_diag"] = kwargs.pop("scale")
        return tfp.distributions.MultivariateNormalDiag(*args, **kwargs)

    return distribution_spec.DistributionSpec(distribution_builder, input_param_spec, sample_spec=sample_spec)


class RiskBudgetActorNetwork(network.DistributionNetwork):
    """Masked-Gaussian actor over `max_pairs` risk-budget logit slots.

    Padding slots (`pair_mask == False`) have loc/scale hard-clamped to fixed
    constants (0, 1) via `tf.where`. Their input features are always exactly
    zero (by `RiskBudgetTrainingEnv`'s construction) and the environment
    ignores whatever the action places there, so left unmasked they'd only
    inject noise into the shared weights' gradient (every step's total
    log_prob sums over ALL slots, including padding, so the PPO surrogate loss
    would otherwise be perturbed by irrelevant padding-slot samples). Clamping
    to constants makes `tf.where`'s gradient to the (masked-out) computed
    branch exactly zero, and makes the value identical for old and new policy
    (ratio contribution exactly 1, entropy contribution a fixed offset) -
    fully decoupling padding slots from training.
    """

    def __init__(self, input_tensor_spec, max_pairs, max_actions, state_feature_dim,
                 input_fc_layer_params=(64,), lstm_size=(32,), output_fc_layer_params=(64,),
                 pair_fc_layer_params=(64, 64), init_action_stddev=0.35, init_means_output_factor=0.1,
                 name="RiskBudgetActorNetwork"):
        global_input_dim = state_feature_dim + max_actions
        lstm_encoder = _shared_global_encoder(
            name, input_fc_layer_params, lstm_size, output_fc_layer_params, global_input_dim)
        output_spec = _mvn_diag_output_spec(max_pairs, name)

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=lstm_encoder.state_spec,
            output_spec=output_spec,
            name=name)

        self._lstm_encoder = lstm_encoder
        self.max_pairs = max_pairs
        self._pair_dense_layers = [
            tf.keras.layers.Dense(units, activation=tf.nn.relu, name=f"{name}_pair_fc_{i}")
            for i, units in enumerate(pair_fc_layer_params)
        ]
        self._mean_layer = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=init_means_output_factor),
            bias_initializer=tf.keras.initializers.Zeros(),
            name=f"{name}_mean")
        std_bias_initializer_value = float(np.log(np.exp(init_action_stddev) - 1.0))
        self._std_bias = tf.Variable(std_bias_initializer_value, dtype=tf.float32, name=f"{name}_std_bias")

    def call(self, observation, step_type, network_state=(), training=False):
        h, network_state = self._lstm_encoder(
            _global_input(observation), step_type=step_type, network_state=network_state, training=training)

        h_per_pair = tf.repeat(h[..., tf.newaxis, :], repeats=self.max_pairs, axis=-2)
        pair_input = tf.concat([
            h_per_pair,
            observation["pair_state_features"],
            observation["pair_action_onehot"],
            observation["pair_prob"][..., tf.newaxis],
            observation["pair_vmin"][..., tf.newaxis],
            observation["pair_vmax"][..., tf.newaxis],
        ], axis=-1)

        x = pair_input
        for layer in self._pair_dense_layers:
            x = layer(x, training=training)
        means = tf.squeeze(self._mean_layer(x, training=training), axis=-1)
        stds = tf.nn.softplus(self._std_bias) * tf.ones_like(means)

        mask = observation["pair_mask"]
        means = tf.where(mask, means, tf.zeros_like(means))
        stds = tf.where(mask, stds, tf.ones_like(stds))

        return self.output_spec.build_distribution(loc=means, scale=stds), network_state


class RiskBudgetValueNetwork(network.Network):
    """Plain scalar value estimate from the same global (state, proposed-action-
    distribution) context used by the actor - mirrors ValueRnnNetwork's own
    simplicity (no per-pair structure needed for a scalar baseline)."""

    def __init__(self, input_tensor_spec, max_actions, state_feature_dim,
                 input_fc_layer_params=(64,), lstm_size=(32,), output_fc_layer_params=(64,),
                 name="RiskBudgetValueNetwork"):
        global_input_dim = state_feature_dim + max_actions
        lstm_encoder = _shared_global_encoder(
            name, input_fc_layer_params, lstm_size, output_fc_layer_params, global_input_dim)
        super().__init__(input_tensor_spec=input_tensor_spec, state_spec=lstm_encoder.state_spec, name=name)
        self._lstm_encoder = lstm_encoder
        self._value_layer = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1),
            name=f"{name}_value")

    def call(self, observation, step_type, network_state=(), training=False):
        h, network_state = self._lstm_encoder(
            _global_input(observation), step_type=step_type, network_state=network_state, training=training)
        value = tf.squeeze(self._value_layer(h, training=training), axis=-1)
        return value, network_state
