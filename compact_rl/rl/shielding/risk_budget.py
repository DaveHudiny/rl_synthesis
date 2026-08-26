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

from compact_rl.rl.shielding.model_info import ModelInfo

State = int
Action = int
Distribution = list[float]
HistoryEntry = Union[State, Distribution, Action]
History = list[HistoryEntry]
BudgetDistribution = dict[tuple[Action, State], float]


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

        reachable_pairs = self._reachable_pairs(state, action_distribution)
        assert reachable_pairs, f"No (action, next-state) pairs reachable from state {state}."

        uniform_prob = 1.0 / len(reachable_pairs)
        return {pair: uniform_prob for pair in reachable_pairs}

    def _reachable_pairs(self, state: State, action_distribution: Distribution) -> list[tuple[Action, State]]:
        transition_matrix = self.model_info.model.transition_matrix
        row_group_start = transition_matrix.get_row_group_start(state)

        pairs = []
        for action, action_prob in enumerate(action_distribution):
            if action_prob <= 0.0:
                continue
            row = transition_matrix.get_row(row_group_start + action)
            for entry in row:
                if entry.value() > 0.0:
                    pairs.append((action, entry.column))
        return pairs


# Registry of risk-budget functions selectable by name (e.g. from the `shielding.py` CLI),
# so new implementations only need to be added here to become available everywhere this is
# used, rather than touching every dispatch site individually.
BUDGET_FUNCTIONS: dict[str, Callable[[ModelInfo], RiskBudgetFunction]] = {
    "uniform": UniformRiskBudget,
}
