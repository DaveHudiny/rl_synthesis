"""Interface and reference implementations for risk distribution functions used by `RiskDistributionShield`.

A risk distribution function determines, at a given point in a trajectory, how the shield's
remaining risk should be distributed across the possible (action, next-state) outcomes of
the current decision. `CapacityRiskDistribution` is the standard one. It is given:
  - `history`: the trajectory so far, as a flat list alternating
    `[state_0, distribution_0, action_0, state_1, distribution_1, action_1, ...]`
    (states and actions are `int`s, distributions are `list[float]`s over actions available
    at the preceding state). By the time a risk distribution function is called for a given step,
    the action actually sampled and played for that step is already the last entry of
    `history` - i.e. `history`'s trailing three entries are `[..., s, d, a]`, so the state
    the query is about is `s = history[-3]`.
  - `action_distribution`: the distribution `d` over actions in effect at `s` (same value as
    `history[-2]`, passed explicitly for convenience).

and must return a distribution over `Act x S` (action, next-state pairs): a
`dict[(action, state), float]` whose support is restricted to pairs `(a, s')` such that
`action_distribution[a] * P(s, a, s') > 0`, where `P` is the transition function of the
underlying MDP.

Implementations are free to use whatever side information they need. The interface itself intentionally has no
dependency on the MDP's transition matrix - only concrete implementations that need
reachability (like `UniformRiskDistribution` below) hold a reference to the model.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union

from compact_rl.rl.shielding.model_info import ModelInfo

State = int
Action = int
Distribution = list[float]
HistoryEntry = Union[State, Distribution, Action]
History = list[HistoryEntry]
RiskDistribution = dict[tuple[Action, State], float]


def reachable_pair_probs(model_info: ModelInfo, state: State, action_distribution: Distribution) -> dict[tuple[Action, State], float]:
    """Raw transition probability P(state, a, s') for every (a, s') pair reachable under
    `action_distribution` (i.e. action_distribution[a] * P(state, a, s') > 0) - not multiplied
    by action_distribution[a], unlike the distribution `RiskDistributionFunction.__call__` returns.
    """
    transition_matrix = model_info.model.transition_matrix
    row_group_start = transition_matrix.get_row_group_start(state)

    probs = {}
    for action, action_prob in enumerate(action_distribution):
        if action_prob <= 0.0:
            continue
        row = transition_matrix.get_row(row_group_start + action)
        for entry in row:
            if entry.value > 0.0:
                probs[(action, entry.column)] = entry.value
    return probs


class RiskDistributionFunction(ABC):
    """Abstract interface for a risk distribution function used by `RiskDistributionShield`."""

    @abstractmethod
    def __call__(self, history: History, action_distribution: Distribution) -> RiskDistribution:
        """Return a distribution over (action, next-state) pairs reachable from the current state.

        Args:
            history: trajectory so far; the current state is `s = history[-3]`.
            action_distribution: the distribution `d` over actions currently in effect at `s`.

        The returned distribution's support must be restricted to pairs `(a, s')` with
        `action_distribution[a] * P(s, a, s') > 0`.
        """
        raise NotImplementedError


class UniformRiskDistribution(RiskDistributionFunction):
    """Spreads the remaining risk uniformly over all (action, next-state) pairs reachable from
    the current state under the current action distribution.

    Unlike the general interface, this implementation needs direct access to the underlying
    MDP's transition matrix (via `ModelInfo`) to determine reachability.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

    def __call__(self, history: History, action_distribution: Distribution) -> RiskDistribution:
        state = history[-3]

        probs = reachable_pair_probs(self.model_info, state, action_distribution)
        assert probs, f"No (action, next-state) pairs reachable from state {state}."

        uniform_prob = 1.0 / len(probs)
        return {pair: uniform_prob for pair in probs}


class HeadroomRiskDistribution(RiskDistributionFunction):
    """Diagnostic heuristic: allocates each reachable pair a share proportional to
    `vmax(s') - vmin(s')` - the same quantity `RiskDistributionShield._make_wasteless`'s own upper
    bound is proportional to (a pair whose successor has vmax == vmin has zero capacity to
    usefully absorb any risk at all, whatever share a risk distribution proposes for it).
    Unlike `CapacityRiskDistribution` it ignores how likely each pair is.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

    def __call__(self, history: History, action_distribution: Distribution) -> RiskDistribution:
        state = history[-3]

        probs = reachable_pair_probs(self.model_info, state, action_distribution)
        assert probs, f"No (action, next-state) pairs reachable from state {state}."

        weights = {pair: self.model_info.vmax[pair[1]] - self.model_info.vmin[pair[1]] for pair in probs}
        total = sum(weights.values())
        if total <= 0:
            uniform_prob = 1.0 / len(probs)
            return {pair: uniform_prob for pair in probs}
        return {pair: w / total for pair, w in weights.items()}


class CapacityRiskDistribution(RiskDistributionFunction):
    """The standard risk distribution: allocates each reachable pair (a, s') a share proportional
    to `action_distribution[a] * P(state, a, s') * (vmax(s') - vmin(s'))` - exactly the
    quantity `_make_wasteless`'s own upper bound is proportional to (weighted reachability
    times headroom capacity), unlike `HeadroomRiskDistribution` which uses headroom alone.

    It is wasteless by construction: the weights sum to qmax - qmin of the played distribution,
    so every successor gets the same fraction slack / (qmax - qmin) <= 1 of its own headroom,
    remaining_risk(s') = vmin(s') + slack / (qmax - qmin) * (vmax(s') - vmin(s')) <= vmax(s'),
    and the wasteless re-projection never changes it.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

    def __call__(self, history: History, action_distribution: Distribution) -> RiskDistribution:
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


class CapacityInformedRiskDistribution(RiskDistributionFunction):
    """`CapacityRiskDistribution` whose capacity comes from known safety values of the policy being shielded instead of the
    worst case `vmax - vmin`.

    `policy_values[s]` is the probability that the UNSHIELDED policy reaches a bad state when it starts in `s`
    (`policy_safety.policy_safety_values` computes it for a memoryless policy, but it may come from anywhere). On top of
    the unavoidable `vmin(s)` the policy itself adds `policy_values[s] - vmin(s)` of risk, so the capacity of a state is

        capacity(s) = trust * max(policy_values[s] - vmin(s), 0) + (1 - trust) * (vmax(s) - vmin(s))

    `trust` = 1 (default) relies on the given values only, `trust` = 0 is `CapacityRiskDistribution` (up to floating point
    rounding), values in between blend the two capacities linearly. The remaining risk is split over the reachable
    pairs (a, s') in proportion to `action_distribution[a] * P(s, a, s') * capacity(s')`, and uniformly if every weight is 0.

    `RiskDistributionShield` picks up the `capacity` method (no other risk distribution has one) and uses `vmin + capacity` in
    place of `vmax` wherever it needs an upper value of a successor: in qmax, and in the usable headroom of the wasteless
    re-projection. In exact arithmetic, with trust = 1 and exact values of a policy that is already safe
    (`policy_values[initial state] <= nu`), every successor then gets exactly the remaining risk the policy itself needs,
    `policy_values[s']`, and that policy is never blocked.
    """

    def __init__(self, model_info: ModelInfo, policy_values, trust: float = 1.0):
        num_states = model_info.model.nr_states
        assert len(policy_values) == num_states, f"policy_values has {len(policy_values)} entries, the model has {num_states} states."
        assert 0.0 <= trust <= 1.0, f"trust must be in [0, 1], got {trust}."
        self.model_info = model_info
        self.policy_values = [float(v) for v in policy_values]
        self.trust = trust
        vmin, vmax = model_info.vmin, model_info.vmax
        self._capacity = [
            trust * max(self.policy_values[s] - vmin[s], 0.0) + (1.0 - trust) * (vmax[s] - vmin[s])
            for s in range(num_states)
        ]

    def capacity(self, state: State) -> float:
        return self._capacity[state]

    def __call__(self, history: History, action_distribution: Distribution) -> RiskDistribution:
        state = history[-3]

        probs = reachable_pair_probs(self.model_info, state, action_distribution)
        assert probs, f"No (action, next-state) pairs reachable from state {state}."

        weights = {(a, s2): action_distribution[a] * p * self._capacity[s2] for (a, s2), p in probs.items()}
        total = sum(weights.values())
        if total <= 0:
            uniform_prob = 1.0 / len(probs)
            return {pair: uniform_prob for pair in probs}
        return {pair: w / total for pair, w in weights.items()}


# Registry of risk distribution functions selectable by name (e.g. from the `shielding.py` CLI),
# so new implementations only need to be added here to become available everywhere this is
# used. `CapacityInformedRiskDistribution` is not included: it needs the policy values as well,
# so `shield_processor.py` constructs it directly.
RISK_DISTRIBUTIONS: dict[str, Callable[[ModelInfo], RiskDistributionFunction]] = {
    "uniform": UniformRiskDistribution,
    "headroom": HeadroomRiskDistribution,
    "capacity": CapacityRiskDistribution,
}
