"""Interface and reference implementations for risk-budget functions used by `ShieldWithBudget`.

A risk-budget function determines, at a given point in a trajectory, how the shield's
remaining risk budget should be split across the possible (action, next-state) outcomes of
the current decision. It is given:
  - `history`: the trajectory so far, as a flat list alternating
    `[state_0, distribution_0, action_0, state_1, distribution_1, action_1, ...]`
    (states and actions are `int`s, distributions are `list[float]`s over actions available
    at the preceding state). By the time a risk-budget function is called for a given step,
    the action actually sampled and played for that step is already the last entry of
    `history` - i.e. `history`'s trailing three entries are `[..., s, d, a]`, so the state
    the query is about is `s = history[-3]`.
  - `action_distribution`: the distribution `d` over actions in effect at `s` (same value as
    `history[-2]`, passed explicitly for convenience).

and must return a distribution over `Act x S` (action, next-state pairs): a
`dict[(action, state), float]` whose support is restricted to pairs `(a, s')` such that
`action_distribution[a] * P(s, a, s') > 0`, where `P` is the transition function of the
underlying MDP.

Implementations may be arbitrarily complex (e.g. backed by a neural network) and are free to
use whatever side information they need. The interface itself intentionally has no
dependency on the MDP's transition matrix - only concrete implementations that need
reachability (like `UniformRiskBudget` below) hold a reference to the model.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

from compact_rl.rl.shielding.model_info import ModelInfo

State = int
Action = int
Distribution = list[float]
HistoryEntry = Union[State, Distribution, Action]
History = list[HistoryEntry]
BudgetDistribution = dict[tuple[Action, State], float]


def reachable_pair_probs(model_info: ModelInfo, state: State, action_distribution: Distribution) -> dict[tuple[Action, State], float]:
    """Raw transition probability P(state, a, s') for every (a, s') pair reachable under
    `action_distribution` (i.e. action_distribution[a] * P(state, a, s') > 0) - not multiplied
    by action_distribution[a], unlike the distribution `RiskBudgetFunction.__call__` returns.
    """
    transition_matrix = model_info.model.transition_matrix
    row_group_start = transition_matrix.get_row_group_start(state)

    probs = {}
    for action, action_prob in enumerate(action_distribution):
        if action_prob <= 0.0:
            continue
        row = transition_matrix.get_row(row_group_start + action)
        for entry in row:
            if entry.value() > 0.0:
                probs[(action, entry.column)] = entry.value()
    return probs


class RiskBudgetFunction(ABC):
    """Abstract interface for a risk-budget function used by `ShieldWithBudget`."""

    @abstractmethod
    def __call__(self, history: History, action_distribution: Distribution) -> BudgetDistribution:
        """Return a distribution over (action, next-state) pairs reachable from the current state.

        Args:
            history: trajectory so far; the current state is `s = history[-3]`.
            action_distribution: the distribution `d` over actions currently in effect at `s`.

        The returned distribution's support must be restricted to pairs `(a, s')` with
        `action_distribution[a] * P(s, a, s') > 0`.
        """
        raise NotImplementedError


class UniformRiskBudget(RiskBudgetFunction):
    """Spreads the risk budget uniformly over all (action, next-state) pairs reachable from
    the current state under the current action distribution.

    Unlike the general interface, this implementation needs direct access to the underlying
    MDP's transition matrix (via `ModelInfo`) to determine reachability.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

    def __call__(self, history: History, action_distribution: Distribution) -> BudgetDistribution:
        state = history[-3]

        probs = reachable_pair_probs(self.model_info, state, action_distribution)
        assert probs, f"No (action, next-state) pairs reachable from state {state}."

        uniform_prob = 1.0 / len(probs)
        return {pair: uniform_prob for pair in probs}


class HeadroomRiskBudget(RiskBudgetFunction):
    """Diagnostic-only heuristic: allocates each reachable pair a share proportional to
    `vmax(s') - vmin(s')` - the same quantity `ShieldWithBudget._make_wasteless`'s own upper
    bound is proportional to (a pair whose successor has vmax == vmin has zero capacity to
    usefully absorb any budget at all, regardless of what any budget function proposes for
    it). Used to check whether a simple, non-learned, headroom-aware proposal can beat
    `UniformRiskBudget` at all under the standard force_wasteless_budget=True evaluation
    setup, where the wasteless re-projection already reshapes ANY proposal toward
    proportional-to-headroom whenever slack is tight - i.e. whether there's a real gap left
    for a smarter proposal (or a learned one) to close, independent of any training algorithm.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

    def __call__(self, history: History, action_distribution: Distribution) -> BudgetDistribution:
        state = history[-3]

        probs = reachable_pair_probs(self.model_info, state, action_distribution)
        assert probs, f"No (action, next-state) pairs reachable from state {state}."

        weights = {pair: self.model_info.vmax[pair[1]] - self.model_info.vmin[pair[1]] for pair in probs}
        total = sum(weights.values())
        if total <= 0:
            uniform_prob = 1.0 / len(probs)
            return {pair: uniform_prob for pair in probs}
        return {pair: w / total for pair, w in weights.items()}


class CapacityRiskBudget(RiskBudgetFunction):
    """Diagnostic-only heuristic: allocates each reachable pair (a, s') a share proportional
    to `action_distribution[a] * P(state, a, s') * (vmax(s') - vmin(s'))` - exactly the
    quantity `_make_wasteless`'s own upper bound is proportional to (weighted reachability
    times headroom capacity), unlike `HeadroomRiskBudget` which uses headroom alone and
    ignores how likely/weighted each pair actually is. Tests whether matching the wasteless
    projection's own natural shape - so it needs little to no correction - does better than
    either uniform or headroom-alone.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

    def __call__(self, history: History, action_distribution: Distribution) -> BudgetDistribution:
        state = history[-3]

        probs = reachable_pair_probs(self.model_info, state, action_distribution)
        assert probs, f"No (action, next-state) pairs reachable from state {state}."

        weights = {
            (a, s2): action_distribution[a] * p * (self.model_info.vmax[s2] - self.model_info.vmin[s2])
            for (a, s2), p in probs.items()
        }
        total = sum(weights.values())
        if total <= 0:
            uniform_prob = 1.0 / len(probs)
            return {pair: uniform_prob for pair in probs}
        return {pair: w / total for pair, w in weights.items()}


class NNRiskBudget(RiskBudgetFunction):
    """Risk-budget function backed by a trained `RiskBudgetActorNetwork`
    (see `risk_budget_networks.py` / `train_risk_budget_shield.py`), used at evaluation time
    to plug a trained network into `ShieldWithBudget`'s normal `correct()` path - the same
    interface every other budget function uses, so it can be evaluated with the same
    methodology (shielding.py, the eval CSV/table).

    The network is recurrent, but `RiskBudgetFunction.__call__` has no notion of a
    persistent per-lane call identity (deliberately - the interface is meant to work for any
    representation, including ones with no state at all). Rather than adding one, this caches
    the LSTM's `network_state` after each call, keyed by `id(history)` - `ShieldWithBudget`
    (and `ShieldProcessor`'s own per-lane tracking) mutate the SAME list object in place across
    consecutive calls within an episode (`history.append(...)`) and only ever replace it with a
    fresh list on episode reset, so `id(history)` is a stable, zero-cost proxy for "same lane,
    same episode, one step later" - without needing any change to the stateless
    `RiskBudgetFunction` interface itself (which is deliberately meant to work for any
    representation, including ones with no state at all).

    On a cache hit (`id(history)` seen before, `num_steps` grew by exactly 1, and the
    episode's first state matches - guards against the vanishingly unlikely case of `id()`
    being reused for an unrelated object) only the ONE new (state, distribution) step is fed
    through the LSTM, using the cached state as `network_state` - O(1) instead of O(episode
    length so far). On a miss (first call for this history, or anything not matching the
    invariants above) it falls back to replaying the full history from scratch exactly as
    before, correctness never depends on the cache being right - a miss just costs the old
    O(episode length) instead of O(1).
    """

    _MAX_CACHE_ENTRIES = 20000

    def __init__(self, model_info: ModelInfo, actor_net, actions: list[str], state_features):
        self.model_info = model_info
        self.actor_net = actor_net
        self.actions = actions
        self.state_features = state_features
        self.max_actions = actor_net.input_tensor_spec["action_distribution"].shape[0]
        # id(history) -> (network_state, num_steps_processed, first_state)
        self._state_cache = {}

    def _pair_features(self, state, action_distribution, feature_dim):
        pairs = list(reachable_pair_probs(self.model_info, state, action_distribution).items())
        assert pairs, f"No (action, next-state) pairs reachable from state {state}."
        max_pairs = self.actor_net.max_pairs

        pair_state_features = np.zeros([max_pairs, feature_dim], dtype=np.float32)
        pair_action_onehot = np.zeros([max_pairs, self.max_actions], dtype=np.float32)
        pair_prob = np.zeros([max_pairs], dtype=np.float32)
        pair_vmin = np.zeros([max_pairs], dtype=np.float32)
        pair_vmax = np.zeros([max_pairs], dtype=np.float32)
        pair_mask = np.zeros([max_pairs], dtype=bool)
        for idx, ((a, s2), p) in enumerate(pairs):
            pair_state_features[idx] = np.asarray(self.state_features[s2], dtype=np.float32)
            pair_action_onehot[idx, a] = 1.0
            pair_prob[idx] = p
            pair_vmin[idx] = self.model_info.vmin[s2]
            pair_vmax[idx] = self.model_info.vmax[s2]
            pair_mask[idx] = True
        return pairs, pair_state_features, pair_action_onehot, pair_prob, pair_vmin, pair_vmax, pair_mask

    def _weights_from_loc(self, loc, pairs):
        n = len(pairs)
        weights = np.exp(loc[:n] - np.max(loc[:n]))
        weights = weights / weights.sum()
        return {pair: float(w) for (pair, _), w in zip(pairs, weights)}

    def _full_replay(self, history, state, action_distribution):
        states_seq = history[0::3]
        distributions_seq = history[1::3]
        num_steps = len(states_seq)

        feature_dim = np.asarray(self.state_features[state]).shape[-1]
        state_features_seq = np.stack(
            [np.asarray(self.state_features[s], dtype=np.float32) for s in states_seq])
        action_distribution_seq = np.zeros([num_steps, self.max_actions], dtype=np.float32)
        for i, d in enumerate(distributions_seq):
            action_distribution_seq[i, :len(d)] = d

        pairs, pair_state_features, pair_action_onehot, pair_prob, pair_vmin, pair_vmax, pair_mask = (
            self._pair_features(state, action_distribution, feature_dim))

        # Only the last time step's per-slot output is ever read (see docstring), but the
        # network processes a full [1, num_steps, ...] sequence at once, so every leaf needs
        # a leading time axis - tile this step's per-slot tensors across it as padding filler
        # for the steps that will be discarded.
        def tile_seq(x):
            return np.broadcast_to(x, (num_steps,) + x.shape).copy()

        observation = {
            "state_features": tf.constant(state_features_seq[np.newaxis, ...]),
            "action_distribution": tf.constant(action_distribution_seq[np.newaxis, ...]),
            "pair_state_features": tf.constant(tile_seq(pair_state_features)[np.newaxis, ...]),
            "pair_action_onehot": tf.constant(tile_seq(pair_action_onehot)[np.newaxis, ...]),
            "pair_prob": tf.constant(tile_seq(pair_prob)[np.newaxis, ...]),
            "pair_vmin": tf.constant(tile_seq(pair_vmin)[np.newaxis, ...]),
            "pair_vmax": tf.constant(tile_seq(pair_vmax)[np.newaxis, ...]),
            "pair_mask": tf.constant(tile_seq(pair_mask)[np.newaxis, ...]),
        }
        step_type = tf.constant(
            [[ts.StepType.FIRST] + [ts.StepType.MID] * (num_steps - 1)], dtype=tf.int32)
        network_state = self.actor_net.get_initial_state(batch_size=1)
        dist, new_network_state = self.actor_net(observation, step_type, network_state, training=False)
        loc = dist.parameters["loc"].numpy()[0, -1]
        return loc, pairs, new_network_state, num_steps, states_seq[0]

    def _incremental_step(self, state, action_distribution, cached_state):
        network_state, _, _ = cached_state
        feature_dim = np.asarray(self.state_features[state]).shape[-1]
        pairs, pair_state_features, pair_action_onehot, pair_prob, pair_vmin, pair_vmax, pair_mask = (
            self._pair_features(state, action_distribution, feature_dim))

        new_state_features = np.asarray(self.state_features[state], dtype=np.float32)
        new_action_distribution = np.zeros([self.max_actions], dtype=np.float32)
        new_action_distribution[:len(action_distribution)] = action_distribution

        observation = {
            "state_features": tf.constant(new_state_features[np.newaxis, np.newaxis, ...]),
            "action_distribution": tf.constant(new_action_distribution[np.newaxis, np.newaxis, ...]),
            "pair_state_features": tf.constant(pair_state_features[np.newaxis, np.newaxis, ...]),
            "pair_action_onehot": tf.constant(pair_action_onehot[np.newaxis, np.newaxis, ...]),
            "pair_prob": tf.constant(pair_prob[np.newaxis, np.newaxis, ...]),
            "pair_vmin": tf.constant(pair_vmin[np.newaxis, np.newaxis, ...]),
            "pair_vmax": tf.constant(pair_vmax[np.newaxis, np.newaxis, ...]),
            "pair_mask": tf.constant(pair_mask[np.newaxis, np.newaxis, ...]),
        }
        step_type = tf.constant([[ts.StepType.MID]], dtype=tf.int32)
        dist, new_network_state = self.actor_net(observation, step_type, network_state, training=False)
        loc = dist.parameters["loc"].numpy()[0, -1]
        return loc, pairs, new_network_state

    def __call__(self, history: History, action_distribution: Distribution) -> BudgetDistribution:
        state = history[-3]
        assert list(history[-2]) == list(action_distribution) or history[-2] == action_distribution

        key = id(history)
        cached = self._state_cache.get(key)
        # history is always complete (state, distribution, action) triples by the time this is
        # called (see class docstring) - len(history)//3 avoids an O(history length) slice just
        # to count steps, which would undo the whole point of the incremental fast path below.
        num_steps = len(history) // 3
        first_state = history[0]

        if cached is not None and cached[1] == num_steps - 1 and cached[2] == first_state:
            loc, pairs, new_network_state = self._incremental_step(state, action_distribution, cached)
        else:
            loc, pairs, new_network_state, num_steps, first_state = self._full_replay(
                history, state, action_distribution)

        if len(self._state_cache) >= self._MAX_CACHE_ENTRIES:
            self._state_cache.clear()
        self._state_cache[key] = (new_network_state, num_steps, first_state)

        return self._weights_from_loc(loc, pairs)


# Registry of risk-budget functions selectable by name (e.g. from the `shielding.py` CLI),
# so new implementations only need to be added here to become available everywhere this is
# used, rather than touching every dispatch site individually. `NNRiskBudget` is deliberately
# not included: it needs more than `model_info` alone (a trained network, state features),
# so `shield_processor.py` constructs it directly instead of through this registry.
BUDGET_FUNCTIONS: dict[str, Callable[[ModelInfo], RiskBudgetFunction]] = {
    "uniform": UniformRiskBudget,
    "headroom": HeadroomRiskBudget,
    "capacity": CapacityRiskBudget,
}
