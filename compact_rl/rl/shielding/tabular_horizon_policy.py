"""A TFPolicy that plays a precomputed, exact finite-horizon-optimal tabular policy - indexed by
(state, remaining-steps), from compute_finite_horizon_policy's own backward-induction DP - through
the same TFPolicy interface every trained neural agent uses elsewhere in this pipeline. This lets
a model-checking-derived policy be shielded/evaluated identically to an RL-trained one (see
ShieldProcessor/shielding.py), for benchmarks where RL exploration alone struggles to find any
reward signal at all but the exact optimum is directly computable instead.
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep

from compact_rl.rl.tools.encoding_methods import observation_and_action_constraint_splitter
from compact_rl.rl.shielding.model_info import ModelInfo


def compute_finite_horizon_policy(model, horizon: int, reward_name: str = "rews",
                                   bad_label: str = "bad", tie_tol: float = 1e-6):
    """Exact backward induction for R{reward_name}max=?[C<=horizon]: V_k(s) = max_a [reward(s,a) +
    sum_s' P(s,a,s') V_{k-1}(s')]. Reward ties are common (e.g. a whole region of the model can
    give 0 reward regardless of action), and a plain argmax breaks them arbitrarily by action
    index - which can silently and consistently prefer an action that walks straight into "bad"
    over an equally reward-optimal one that doesn't, since nothing in the reward signal itself
    penalizes that. So this jointly tracks, for each state/step, a SECOND value B_k(s) = the min
    P[F bad] achievable AMONG the reward-maximizing actions only (using the previous step's
    B_{k-1}, itself computed the same lexicographic way) - reward is still maximized first and
    exactly, but among whatever ties that leaves, the safest of them is chosen, rather than
    whichever happens to come first by index. Returns (policy_table, value_at_initial_state)
    where policy_table[s][k] is a local action index (0-indexed into that state's own available
    choices - NOT a global action index), defined for k in [0, horizon] (k=0 unused as a lookup
    target but kept for uniform indexing). O(states * actions * successors * horizon) - tractable
    for these model sizes (seconds, not minutes, even at ~9000 states / 100 steps)."""
    n_states = model.nr_states
    tm = model.transition_matrix
    rm = model.reward_models[reward_name]
    bad_states = set(model.labeling.get_states(bad_label))

    # A model with only state rewards pays them on arrival, once, as the simulator does (an absorbing
    # state ends the episode) - not on every step spent there as Storm's C<=N would count them.
    arrival_rewards = np.asarray(rm.state_rewards) if rm.has_state_rewards and not rm.has_state_action_rewards else None

    actions_per_state = []
    for s in range(n_states):
        n_a = model.get_nr_available_actions(s)
        start = tm.get_row_group_start(s)
        acts = []
        for a in range(n_a):
            row = tm.get_row(start + a)
            succs = [(e.column, e.value()) for e in row]
            if rm.has_state_action_rewards:
                rew = rm.get_state_action_reward(start + a)
            elif arrival_rewards is not None and not all(s2 == s for s2, _ in succs):
                rew = sum(p * arrival_rewards[s2] for s2, p in succs)
            else:
                rew = 0.0
            acts.append((succs, rew))
        actions_per_state.append(acts)

    V = np.zeros(n_states)
    B = np.array([1.0 if s in bad_states else 0.0 for s in range(n_states)])
    policy_table = [[0] * (horizon + 1) for _ in range(n_states)]
    for k in range(1, horizon + 1):
        V_new = np.zeros(n_states)
        B_new = np.zeros(n_states)
        for s in range(n_states):
            if s in bad_states:
                V_new[s], B_new[s] = V[s], 1.0
                continue
            if not actions_per_state[s]:
                V_new[s], B_new[s] = V[s], B[s]
                continue
            best_val = max(rew + sum(p * V[s2] for s2, p in succs)
                            for succs, rew in actions_per_state[s])
            best_a, best_bad = 0, np.inf
            for a, (succs, rew) in enumerate(actions_per_state[s]):
                val = rew + sum(p * V[s2] for s2, p in succs)
                if val < best_val - tie_tol:
                    continue  # not reward-optimal, never a candidate regardless of safety
                bad_prob = sum(p * B[s2] for s2, p in succs)
                if bad_prob < best_bad:
                    best_bad, best_a = bad_prob, a
            V_new[s] = best_val
            B_new[s] = best_bad
            policy_table[s][k] = best_a
        V, B = V_new, B_new

    s0 = list(model.initial_states)[0]
    return policy_table, float(V[s0])


def compute_bad_probability(model, policy_table, horizon: int, bad_label: str = "bad",
                             noise_epsilon: float = 0.0):
    """P=?[F bad] under the (state, remaining-steps)-indexed policy_table, exact backward DP -
    a direct check of whether a reward-optimal policy is safe enough to use before ever running
    it, without needing the shield at all. noise_epsilon > 0 accounts for the SAME epsilon-greedy
    mixing TabularHorizonPolicy itself applies (1 - noise_epsilon on the table's optimal action,
    noise_epsilon spread uniformly over that state's other actions) - computed exactly here, not
    by simulation, so the right epsilon for a target bad-probability can be found by a fast
    search instead of repeated noisy rollouts."""
    n_states = model.nr_states
    tm = model.transition_matrix
    bad_states = set(model.labeling.get_states(bad_label))

    actions_per_state = []
    for s in range(n_states):
        n_a = model.get_nr_available_actions(s)
        start = tm.get_row_group_start(s)
        acts = []
        for a in range(n_a):
            row = tm.get_row(start + a)
            acts.append([(e.column, e.value()) for e in row])
        actions_per_state.append(acts)

    Q = np.array([1.0 if s in bad_states else 0.0 for s in range(n_states)])
    for k in range(1, horizon + 1):
        Q_new = np.zeros(n_states)
        for s in range(n_states):
            if s in bad_states:
                Q_new[s] = 1.0
                continue
            acts = actions_per_state[s]
            if not acts:
                Q_new[s] = Q[s]
                continue
            opt_a = policy_table[s][k]
            if noise_epsilon <= 0 or len(acts) < 2:
                succs = acts[opt_a]
                Q_new[s] = sum(p * Q[s2] for s2, p in succs)
            else:
                other_weight = noise_epsilon / (len(acts) - 1)
                total = 0.0
                for a, succs in enumerate(acts):
                    w = (1.0 - noise_epsilon) if a == opt_a else other_weight
                    total += w * sum(p * Q[s2] for s2, p in succs)
                Q_new[s] = total
        Q = Q_new
    s0 = list(model.initial_states)[0]
    return float(Q[s0])


class TabularHorizonPolicy(TFPolicy):
    """policy_state is a per-lane int32 "remaining steps" counter, reset to `horizon` whenever
    time_step.is_first() (mirrors how a recurrent policy's own state gets reset at episode
    boundaries elsewhere in this pipeline - see e.g. RiskBudgetTrainingEnv's just_reset handling).
    """

    def __init__(self, action_spec, time_step_spec, model_info: ModelInfo, actions: list,
                 policy_table, horizon: int, noise_epsilon: float = 0.1):
        super().__init__(time_step_spec, action_spec,
                          policy_state_spec=tf.TensorSpec([], tf.int32),
                          observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self._model_info = model_info
        self._actions = actions
        self._policy_table = policy_table
        self._horizon = horizon
        # A hard one-hot here would give --deterministic-agent's "off" (stochastic) mode nothing
        # to actually sample from - a real trained policy's softmax is never perfectly one-hot
        # either. noise_epsilon of the mass is spread uniformly over this state's OTHER available
        # actions, keeping (1 - noise_epsilon) on the computed-optimal one; --deterministic-agent
        # still recovers the same optimal action via argmax regardless of this spread.
        self._noise_epsilon = noise_epsilon

    def _get_initial_state(self, batch_size):
        return tf.fill([batch_size], self._horizon)

    def _choice_labels(self, state):
        model = self._model_info.model
        start = model.transition_matrix.get_row_group_start(state)
        end = model.transition_matrix.get_row_group_end(state)
        return [model.choice_labeling.get_labels_of_choice(c).pop() for c in range(start, end)]

    def _distribution(self, time_step, policy_state):
        observation, mask = self.observation_and_action_constraint_splitter(time_step.observation)
        integers = time_step.observation["integer"].numpy()
        is_first = time_step.is_first().numpy()
        prev_remaining = policy_state.numpy() if policy_state is not None and len(policy_state.shape) > 0 else np.full([integers.shape[0]], self._horizon)
        remaining = np.where(is_first, self._horizon, prev_remaining)

        batch_size = integers.shape[0]
        num_actions = len(self._actions)
        probs = np.zeros((batch_size, num_actions), dtype=np.float32)
        eps = self._noise_epsilon
        for i in range(batch_size):
            state = self._model_info.observation_to_state[int(integers[i, 0])]
            k = int(np.clip(remaining[i], 0, self._horizon))
            local_action = self._policy_table[state][k]
            choice_labels = self._choice_labels(state)
            action_label = choice_labels[local_action]
            global_action = self._actions.index(action_label)
            other_labels = [l for j, l in enumerate(choice_labels) if j != local_action]
            if other_labels and eps > 0:
                probs[i, global_action] = 1.0 - eps
                share = eps / len(other_labels)
                for l in other_labels:
                    probs[i, self._actions.index(l)] += share
            else:
                probs[i, global_action] = 1.0

        next_state = tf.constant(remaining - 1, dtype=tf.int32)
        policy_step = PolicyStep(
            action=tfp.distributions.Categorical(logits=tf.math.log(tf.constant(probs) + 1e-10)),
            state=next_state)
        return policy_step

    def _action(self, time_step, policy_state, seed=None):
        step = self._distribution(time_step, policy_state)
        action = tf.argmax(step.action.logits, axis=-1, output_type=tf.int32)
        return PolicyStep(action=action, state=step.state)
