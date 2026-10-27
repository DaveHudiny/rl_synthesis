"""Exact safety values of a memoryless policy, the input `CapacityInformedRiskBudget` needs."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def policy_safety_values(model, state_distributions, bad_label: str = "bad") -> np.ndarray:
    """Probability of ever reaching a bad state, from every state, when the unshielded policy plays
    `state_distributions[s]` in state s (a distribution over the LOCAL actions of s, in the order of the state's choices,
    like the distributions the shields receive). Bad states have value 1. The value is the solution of one sparse linear
    system on the states from which a bad state is reachable at all; every other state has value 0 (a run that can
    never reach bad, in particular one that is absorbed elsewhere, has no risk)."""
    n = model.nr_states
    transition_matrix = model.transition_matrix
    rows, cols, vals = [], [], []
    for state in range(n):
        start = transition_matrix.get_row_group_start(state)
        for action, prob in enumerate(state_distributions[state]):
            if prob <= 0.0:
                continue
            for entry in transition_matrix.get_row(start + action):
                rows.append(state)
                cols.append(entry.column)
                vals.append(prob * entry.value)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    bad = np.zeros(n, dtype=bool)
    bad[list(model.labeling.get_states(bad_label))] = True
    not_bad = ~bad
    into_bad = P @ bad.astype(float)  # one-step probability of entering a bad state
    can_reach = (into_bad > 0) & not_bad
    while True:
        grow = ((P @ can_reach.astype(float)) > 0) & not_bad & ~can_reach
        if not grow.any():
            break
        can_reach |= grow

    values = np.zeros(n)
    values[bad] = 1.0
    transient = np.flatnonzero(can_reach)
    if len(transient):
        system = sp.identity(len(transient), format="csc") - P[transient][:, transient].tocsc()
        values[transient] = spla.spsolve(system, into_bad[transient])
    # the system can be nearly singular (a run that lingers for a very long time), and the solve then overshoots
    # 1 by up to about 1e-3; these are probabilities
    return np.clip(values, 0.0, 1.0)
