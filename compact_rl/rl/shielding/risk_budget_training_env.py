"""A custom TF-Agents PyEnvironment for training a learned risk-budget function via PPO.

Wraps [an EnvironmentWrapperVec simulating the underlying MDP] + [a fixed, already-loaded
policy] + [ShieldWithBudget-style risk-budget bookkeeping and blocking-correction], exposing
the risk-budget allocation itself as the RL action to be trained.

At each decision point, the observation given to the RL policy describes the current state,
the action distribution currently in effect there (already shield-corrected), and the set of
(action, next-state) pairs reachable under it - padded/masked to a fixed `max_pairs` (bounded
by the model's branching factor, not its size, so this stays small even for large models). The
action is a `max_pairs`-dimensional vector of logits; only the valid slots (per this step's
`pair_mask`) are used, softmax-normalized internally to form the actual risk-budget
distribution handed to `ShieldWithBudget`'s own bookkeeping methods.

The reward returned for an action is `-D`, the L1 distance between the fixed policy's next raw
proposal and what the shield - using that action's risk-budget choice to update
`remaining_risk` - actually allows. This is realized one decision later: the risk-budget chosen
when arriving at state s determines `remaining_risk` for whatever successor state s' is
realized, and D is only known once the policy's proposal at s' has been checked against it.
This directly implements the intervention-cost objective
    J_gamma(b; pi) = E_tau[sum_t gamma^t * D(pi(tau(t)), shield_b(pi, tau(t)))]
by returning `discount=gamma` on every non-terminal step (0 on terminal ones) and letting the
standard PPO/GAE machinery handle the discounted-return accounting - no manual per-episode
discount tracking needed here.
"""

import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import time_step_spec as build_time_step_spec

from compact_rl.rl.environment import py_environment
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.shielding.model_info import ModelInfo
from compact_rl.rl.shielding.risk_budget import UniformRiskBudget, reachable_pair_probs
from compact_rl.rl.shielding.shields import ShieldWithBudget, clamp_distribution


def compute_max_pairs(model_info: ModelInfo) -> tuple[int, int]:
    """(max_actions, max_branching) over every state in the model - the tightest fixed bound on
    how many (action, next-state) pairs can ever be reachable from a single state, needed for a
    fixed-shape PPO action/observation spec. Bounded by branching factor, not model size."""
    model = model_info.model
    max_actions = 0
    max_branching = 0
    for state in range(model.nr_states):
        actions_count = model.get_nr_available_actions(state)
        max_actions = max(max_actions, actions_count)
        row_group_start = model.transition_matrix.get_row_group_start(state)
        for action in range(actions_count):
            row = model.transition_matrix.get_row(row_group_start + action)
            branching = sum(1 for entry in row if entry.value() > 0.0)
            max_branching = max(max_branching, branching)
    return max_actions, max_branching


class RiskBudgetTrainingEnv(py_environment.PyEnvironment):
    def __init__(self, environment: EnvironmentWrapperVec, policy, model_info: ModelInfo,
                 actions: list, nu: float, gamma: float = 0.99, force_wasteless_budget: bool = True,
                 use_l1_projection: bool = True):
        super().__init__()
        self.num_envs = environment.num_envs
        self._environment = environment
        self._policy = policy
        self.model_info = model_info
        self.actions = actions
        self.nu = nu
        self.gamma = gamma
        self.force_wasteless_budget = force_wasteless_budget

        self.max_actions, self.max_branching = compute_max_pairs(model_info)
        self.max_pairs = self.max_actions * self.max_branching
        self._state_feature_dim = int(environment.observation_valuations.shape[-1])

        self._obs_spec = {
            "state_features": tensor_spec.TensorSpec([self._state_feature_dim], tf.float32, "state_features"),
            "action_distribution": tensor_spec.TensorSpec([self.max_actions], tf.float32, "action_distribution"),
            "pair_state_features": tensor_spec.TensorSpec([self.max_pairs, self._state_feature_dim], tf.float32, "pair_state_features"),
            "pair_action_onehot": tensor_spec.TensorSpec([self.max_pairs, self.max_actions], tf.float32, "pair_action_onehot"),
            "pair_prob": tensor_spec.TensorSpec([self.max_pairs], tf.float32, "pair_prob"),
            "pair_vmin": tensor_spec.TensorSpec([self.max_pairs], tf.float32, "pair_vmin"),
            "pair_vmax": tensor_spec.TensorSpec([self.max_pairs], tf.float32, "pair_vmax"),
            "pair_mask": tensor_spec.TensorSpec([self.max_pairs], tf.bool, "pair_mask"),
        }
        self._action_spec = tensor_spec.TensorSpec([self.max_pairs], tf.float32, "risk_budget_logits")
        self._time_step_spec = build_time_step_spec(
            observation_spec=self._obs_spec,
            reward_spec=tensor_spec.TensorSpec((), tf.float32, "reward"))

        # Reuse ShieldWithBudget's own stateless-given-model_info bookkeeping helpers
        # (_qmin/_qmax/_transition_prob/_closest_allowed_distribution/_make_wasteless) without
        # going through its own trace-indexed correct()/budget-callback loop - self.budget on
        # this instance is never actually invoked; the RL action stands in for its result.
        self._shield = ShieldWithBudget(model_info=model_info, actions=actions, nu=nu,
                                         budget=UniformRiskBudget(model_info),
                                         force_wasteless_budget=force_wasteless_budget,
                                         use_l1_projection=use_l1_projection)

        # Per-lane bookkeeping (mirrors ShieldWithBudget's own trace_index-indexed lists).
        self._remaining_risk = [0.0] * self.num_envs
        self._last_states = [None] * self.num_envs
        self._last_distributions = [None] * self.num_envs
        self._last_qmin_ds = [None] * self.num_envs
        # Cached from the most recent observation-building call - needed by _step to interpret
        # the RL action's slots and to compute this step's realized risk-budget share.
        self._pair_actions = [None] * self.num_envs  # list[int], local action index per slot
        self._pair_states = [None] * self.num_envs   # list[int], next-state index per slot
        self._num_valid_pairs = [0] * self.num_envs
        self._policy_state = None
        # Diagnostic only (not used by reset/step's own logic): the risk_budget_distribution
        # share the RL action actually placed on the (action, next_state) pair that ended up
        # being realized this step - i.e. the only part of that step's action that had any
        # effect on the environment. None on a step where the lane just auto-reset (no
        # risk-budget update happens then). Lets external callers correlate "share given to
        # the realized pair" against the resulting reward without re-deriving it.
        self._last_used_risk_budget_share = [None] * self.num_envs
        # Diagnostic only: the actual additive boost to remaining_risk this step's chosen
        # share bought at the realized pair (risk_budget * slack / transition_prob) - unlike
        # the raw share above, this is the mechanistically comparable quantity across
        # different (state, context) pairs, since the same share can buy very different
        # amounts of headroom depending on that pair's slack/transition_prob.
        self._last_additive_boost = [None] * self.num_envs

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return self._time_step_spec

    def _choice_labels(self, state):
        model = self.model_info.model
        row_group_start = model.transition_matrix.get_row_group_start(state)
        row_group_end = model.transition_matrix.get_row_group_end(state)
        return [model.choice_labeling.get_labels_of_choice(c).pop() for c in range(row_group_start, row_group_end)]

    def _map_to_local(self, state, global_probs):
        # global_probs: [nr_global_actions] array (e.g. softmax over self.actions) -> local,
        # per-state distribution matching get_nr_available_actions(state) - same convention as
        # ShieldProcessor.map_played_distribution.
        choice_labels = self._choice_labels(state)
        local = [float(global_probs[self.actions.index(label)]) for label in choice_labels]
        total = sum(local)
        if total > 0:
            return [p / total for p in local]
        return [1.0 / len(local)] * len(local)

    def _query_policy(self, time_step):
        policy_step = self._policy.distribution(time_step, self._policy_state)
        self._policy_state = policy_step.state
        logits = policy_step.action.logits.numpy()
        return tf.nn.softmax(logits).numpy()

    def _build_observation_for_lane(self, i, state, distribution):
        probs = reachable_pair_probs(self.model_info, state, distribution)
        pairs = list(probs.items())  # [((a, s2), p), ...]
        assert len(pairs) <= self.max_pairs, (
            f"state {state} has {len(pairs)} reachable pairs, exceeding max_pairs={self.max_pairs} "
            "computed from the model - compute_max_pairs must be wrong.")

        self._pair_actions[i] = [a for (a, _), _ in pairs]
        self._pair_states[i] = [s2 for (_, s2), _ in pairs]
        self._num_valid_pairs[i] = len(pairs)

        state_features = np.asarray(self._environment.observation_valuations[state], dtype=np.float32)
        action_distribution = np.zeros([self.max_actions], dtype=np.float32)
        action_distribution[:len(distribution)] = distribution

        pair_state_features = np.zeros([self.max_pairs, self._state_feature_dim], dtype=np.float32)
        pair_action_onehot = np.zeros([self.max_pairs, self.max_actions], dtype=np.float32)
        pair_prob = np.zeros([self.max_pairs], dtype=np.float32)
        pair_vmin = np.zeros([self.max_pairs], dtype=np.float32)
        pair_vmax = np.zeros([self.max_pairs], dtype=np.float32)
        pair_mask = np.zeros([self.max_pairs], dtype=bool)
        for idx, ((a, s2), p) in enumerate(pairs):
            pair_state_features[idx] = np.asarray(self._environment.observation_valuations[s2], dtype=np.float32)
            pair_action_onehot[idx, a] = 1.0
            pair_prob[idx] = p
            pair_vmin[idx] = self.model_info.vmin[s2]
            pair_vmax[idx] = self.model_info.vmax[s2]
            pair_mask[idx] = True

        return {
            "state_features": state_features,
            "action_distribution": action_distribution,
            "pair_state_features": pair_state_features,
            "pair_action_onehot": pair_action_onehot,
            "pair_prob": pair_prob,
            "pair_vmin": pair_vmin,
            "pair_vmax": pair_vmax,
            "pair_mask": pair_mask,
        }

    def _stack_observations(self, per_lane_obs):
        return {key: tf.constant(np.stack([obs[key] for obs in per_lane_obs])) for key in per_lane_obs[0]}

    def _correct_distribution(self, state, local_proposed, remaining_risk):
        """Mirrors ShieldWithBudget.correct's own use_l1_projection branch (shields.py) - the
        L1-closest-allowed projection by default, or the older clamp-to-vmin-safe correction
        when training explicitly for the clamp-mode shield."""
        if self._shield.use_l1_projection:
            return self._shield._closest_allowed_distribution(state, local_proposed, remaining_risk)
        return clamp_distribution(local_proposed, self._shield.vmin_actions[state])

    def _reset(self):
        time_step = self._environment.reset()
        states = np.asarray(self._environment.current_state().vertices)
        self._policy_state = self._policy.get_initial_state(batch_size=self.num_envs)
        proposed_probs = self._query_policy(time_step)

        per_lane_obs = []
        for i in range(self.num_envs):
            state = int(states[i])
            local_proposed = self._map_to_local(state, proposed_probs[i])
            remaining_risk = min(self.nu, self.model_info.vmax[state])
            qmin_d = self._shield._qmin(state, local_proposed)
            if qmin_d > remaining_risk:
                output_distribution = self._correct_distribution(state, local_proposed, remaining_risk)
                qmin_d = self._shield._qmin(state, output_distribution)
            else:
                output_distribution = local_proposed

            self._remaining_risk[i] = remaining_risk
            self._last_states[i] = state
            self._last_distributions[i] = output_distribution
            self._last_qmin_ds[i] = qmin_d

            per_lane_obs.append(self._build_observation_for_lane(i, state, output_distribution))

        return ts.TimeStep(
            step_type=tf.fill([self.num_envs], ts.StepType.FIRST),
            reward=tf.zeros([self.num_envs], dtype=tf.float32),
            discount=tf.ones([self.num_envs], dtype=tf.float32),
            observation=self._stack_observations(per_lane_obs))

    def _step(self, action):
        action = np.asarray(action)  # [num_envs, max_pairs] logits

        # 1. qmax/slack depend only on already-known quantities (last state/distribution/qmin),
        #    not on this step's RL action - compute first, matching ShieldWithBudget.correct's
        #    own order, before the risk-budget distribution (this step's RL action) is used.
        qmaxes = [self._shield._qmax(self._last_states[i], self._last_distributions[i]) for i in range(self.num_envs)]
        slacks = [min(self._remaining_risk[i], qmaxes[i]) - self._last_qmin_ds[i] for i in range(self.num_envs)]

        global_actions = np.zeros([self.num_envs], dtype=np.int32)
        local_last_actions = [0] * self.num_envs
        pending_risk_budget = [None] * self.num_envs
        for i in range(self.num_envs):
            n = self._num_valid_pairs[i]
            logits = action[i, :n]
            weights = np.exp(logits - np.max(logits))
            weights = weights / weights.sum()
            risk_budget_distribution = {
                (self._pair_actions[i][k], self._pair_states[i][k]): float(weights[k]) for k in range(n)
            }
            if self.force_wasteless_budget:
                risk_budget_distribution = self._shield._make_wasteless(
                    risk_budget_distribution, self._last_states[i], self._last_distributions[i], slacks[i])
            pending_risk_budget[i] = risk_budget_distribution

            # Sample the action actually played from the (already shield-corrected) distribution
            # decided for this lane's current state on the previous _reset()/_step() call.
            local_action = int(np.random.choice(len(self._last_distributions[i]), p=self._last_distributions[i]))
            local_last_actions[i] = local_action
            action_label = self._choice_labels(self._last_states[i])[local_action]
            global_actions[i] = self.actions.index(action_label)

        time_step = self._environment.step(tf.constant(global_actions))
        new_states = np.asarray(self._environment.current_state().vertices)
        step_types = np.asarray(time_step.step_type)
        proposed_probs = self._query_policy(time_step)

        rewards = np.zeros([self.num_envs], dtype=np.float32)
        discounts = np.zeros([self.num_envs], dtype=np.float32)
        per_lane_obs = []
        for i in range(self.num_envs):
            new_state = int(new_states[i])
            just_reset = step_types[i] == ts.StepType.FIRST

            if just_reset:
                # The underlying env auto-reset this lane - the "transition" from the old
                # episode's last state to this new episode's initial state isn't a real MDP
                # transition, so (matching ShieldWithBudget.correct's own `reset` branch) skip
                # the risk-budget-based update entirely.
                self._remaining_risk[i] = min(self.nu, self.model_info.vmax[new_state])
                self._last_used_risk_budget_share[i] = None
                self._last_additive_boost[i] = None
            else:
                transition_prob = self._shield._transition_prob(
                    self._last_states[i], self._last_distributions[i], local_last_actions[i], new_state)
                assert transition_prob > 0, (
                    f"Transition probability is zero for last_state {self._last_states[i]}, "
                    f"last_action {local_last_actions[i]}, current_state {new_state}.")
                risk_budget = pending_risk_budget[i].get((local_last_actions[i], new_state), 0.0)
                additive_boost = (risk_budget * slacks[i]) / transition_prob
                self._remaining_risk[i] = self.model_info.vmin[new_state] + additive_boost
                self._last_used_risk_budget_share[i] = risk_budget
                self._last_additive_boost[i] = additive_boost

            local_proposed = self._map_to_local(new_state, proposed_probs[i])
            qmin_d = self._shield._qmin(new_state, local_proposed)
            if qmin_d > self._remaining_risk[i]:
                output_distribution = self._correct_distribution(new_state, local_proposed, self._remaining_risk[i])
                qmin_d = self._shield._qmin(new_state, output_distribution)
            else:
                output_distribution = local_proposed

            # The reward for the RL action chosen THIS call is only realized now, one decision
            # later (see module docstring) - D measures how much the just-realized transition's
            # remaining_risk forced the shield to deviate from the fixed policy's own proposal.
            rewards[i] = -sum(abs(p - q) for p, q in zip(local_proposed, output_distribution))
            discounts[i] = 0.0 if just_reset else self.gamma

            self._last_states[i] = new_state
            self._last_distributions[i] = output_distribution
            self._last_qmin_ds[i] = qmin_d

            per_lane_obs.append(self._build_observation_for_lane(i, new_state, output_distribution))

        return ts.TimeStep(
            step_type=tf.constant(step_types),
            reward=tf.constant(rewards),
            discount=tf.constant(discounts),
            observation=self._stack_observations(per_lane_obs))
