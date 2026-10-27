from compact_rl.rl.tools.args_emulator import ArgsEmulator
import tensorflow as tf

from compact_rl.rl.shielding.model_info import ModelInfo
import compact_rl.rl.shielding.shields
from compact_rl.rl.shielding.constructed_shield_data import ShieldData
from compact_rl.rl.shielding.risk_distribution import RISK_DISTRIBUTIONS, CapacityInformedRiskDistribution

import stormpy
import numpy as np

import pickle
import math


class _RunningStat:
    """Online (Welford) mean/variance accumulator - avoids keeping a raw per-episode list in
    memory, which matters given this is meant to run over many thousands of episodes under a
    tight memory budget."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        return math.sqrt(self._m2 / self.n)


class ShieldProcessor:
    def __init__(self, actions : list[str], model : stormpy.storage.SparsePomdp, nu : float, shield_type : str, args : ArgsEmulator = None, shield_memory : int = 0, debug: bool = False, shield_folder: str = None, deterministic_agent: bool = False, risk_distribution: str = "capacity", use_clamp: bool = False, force_wasteless_risk_distribution: bool = True, discount_factor: float = 0.99, policy_values=None, informed_trust: float = 1.0, blocking_lambda: float = None):
        self.args = args
        self.actions = actions
        self.shield_folder = shield_folder
        self.deterministic_agent = deterministic_agent

        assert model.nr_states == model.nr_observations, "We currently only support shielding for MDPs."
        assert model.initial_states is not None and len(model.initial_states) == 1, "We currently only support single initial state models."

        components = stormpy.SparseModelComponents(transition_matrix=model.transition_matrix,
                                                  reward_models=model.reward_models,
                                                  state_labeling=model.labeling)

        components.choice_labeling = model.choice_labeling
        if model.has_state_valuations():
            components.state_valuations = model.state_valuations
        if model.has_choice_origins():
            components.choice_origins = model.choice_origins
        
        mdp = stormpy.storage.SparseMdp(components)

        # get Vmin and Vmax values for all states
        min_formula = stormpy.parse_properties("Pmin=? [ F \"bad\" ]")
        max_formula = stormpy.parse_properties("Pmax=? [ F \"bad\" ]")
        min_result = stormpy.model_checking(mdp, min_formula[0])
        max_result = stormpy.model_checking(mdp, max_formula[0])
        vmin = min_result.get_values()
        vmax = max_result.get_values()
        # Print vmin and vmax for the initial state
        print("Vmin and Vmax for initial state:", vmin[mdp.initial_states[0]], vmax[mdp.initial_states[0]])

        self.bad_states = list(mdp.labeling.get_states("bad"))

        # model checking results for debugging
        if debug:
            print(model)
            if "goal" in model.labeling.get_labels():
                reach_formula = stormpy.parse_properties("Pmax=? [ F \"goal\" ]")
                until_formula = stormpy.parse_properties("Pmax=? [ !\"bad\" U \"goal\" ]")
                goal_formula = stormpy.parse_properties("Pmax=? [ F \"goal\" ]")
                reach_result = stormpy.model_checking(mdp, reach_formula[0])
                until_result = stormpy.model_checking(mdp, until_formula[0])
                goal_result = stormpy.model_checking(mdp, goal_formula[0])
                print("Max reachability probabilities to goal from initial state:", reach_result.get_values()[mdp.initial_states[0]])
                print("Max until probabilities to goal from initial state:", until_result.get_values()[mdp.initial_states[0]])
            # print(vmin)
            # print(self.bad_states)
            reward_formula = stormpy.parse_properties("Rmax=? [ C<=100 ]")
            reward_result = stormpy.model_checking(mdp, reward_formula[0])
            print(f"Vmin {vmin[mdp.initial_states[0]]}, Vmax {vmax[mdp.initial_states[0]]}")
            print("Max expected rewards from initial state:", reward_result.get_values()[mdp.initial_states[0]])

            # win_formula = stormpy.parse_properties("Pmax=? [ F \"win\" ]")
            # win_result = stormpy.model_checking(mdp, win_formula[0])
            # print("Max reachability probabilities to win from initial state:", win_result.get_values()[mdp.initial_states[0]])

            # min_lose_formula = stormpy.parse_properties("Pmin=? [ F \"lose\" | \"bad\" ]")
            # min_lose_result = stormpy.model_checking(mdp, min_lose_formula[0])
            # print("Min reachability probabilities to lose from initial state:", min_lose_result.get_values()[mdp.initial_states[0]])
            exit()

        observation_to_state = [None] * model.nr_observations
        for state in range(model.nr_states):
            obs = model.get_observation(state)
            observation_to_state[obs] = state

        assert None not in observation_to_state, "Some observations do not map to any state."
            
        model_info = ModelInfo(model=model, observation_to_state=observation_to_state, bad_state="bad", vmin=vmin, vmax=vmax)

        if shield_type == 'identity':
            self.shield = compact_rl.rl.shielding.shields.IdentityShield(model_info=model_info, actions=self.actions)
        elif shield_type == 'standard':
            self.shield = compact_rl.rl.shielding.shields.StandardShield(model_info=model_info, actions=self.actions)
        elif shield_type == 'pessimistic':
            self.shield = compact_rl.rl.shielding.shields.PessimisticShield(model_info=model_info, actions=self.actions, nu=nu)
        elif shield_type == 'optimistic':
            self.shield = compact_rl.rl.shielding.shields.OptimisticShield(model_info=model_info, actions=self.actions, nu=nu)
        elif shield_type == 'delta':
            self.shield = compact_rl.rl.shielding.shields.DeltaShield(model_info=model_info, actions=self.actions, delta=nu)
        elif shield_type == 'self-constructing-static':
            self.shield = compact_rl.rl.shielding.shields.SelfConstructingShield(model_info=model_info, actions=self.actions, nu=nu, memory=shield_memory)
        elif shield_type == 'self-constructing-safe':
            self.shield = compact_rl.rl.shielding.shields.SelfConstructingShieldOnline(model_info=model_info, actions=self.actions, nu=nu, memory=shield_memory)
        elif shield_type == 'self-constructing-unsafe':
            self.shield = compact_rl.rl.shielding.shields.SelfConstructingShieldOffline(model_info=model_info, actions=self.actions, nu=nu, memory=shield_memory)
        elif shield_type == 'risk-distribution':
            if risk_distribution == 'capacity-informed':
                assert policy_values is not None, "risk_distribution='capacity-informed' requires policy_values (the safety value of the shielded policy in every state)."
                risk_distribution_fn = CapacityInformedRiskDistribution(model_info, policy_values, informed_trust)
            else:
                risk_distribution_fn = RISK_DISTRIBUTIONS[risk_distribution](model_info)
            self.shield = compact_rl.rl.shielding.shields.RiskDistributionShield(model_info=model_info, actions=self.actions, nu=nu, risk_distribution=risk_distribution_fn, use_l1_projection=not use_clamp, force_wasteless_risk_distribution=force_wasteless_risk_distribution, discount_factor=discount_factor, blocking_lambda=blocking_lambda)
        else:
            raise ValueError(f"Unknown shield type: {shield_type}")
        
        self.shield.rounding_precision = 6

        # --- Per-episode metric tracking (safety, allowed-actions ratio, discounted
        # intervention cost, time-to-first-block, whether-anything-was-blocked) ---
        # Lives here (rather than inside a Shield subclass) because "current_state is bad" and
        # episode boundaries ("resets[i]") are both already visible at this level for every
        # shield type, and because "bad" states bypass shield.correct() entirely (see the
        # goal/fail branch below), so a Shield subclass alone could never observe them.
        self.discount_factor = discount_factor
        self._bad_states_set = frozenset(self.bad_states)
        self._episode_started = []
        self._episode_bad = []
        self._episode_steps = []
        self._episode_blocked = []
        self._episode_first_block_step = []
        self._episode_cost = []
        self.metric_safety = _RunningStat()
        self.metric_allowed_ratio = _RunningStat()
        self.metric_discounted_cost = _RunningStat()
        self.metric_first_block_step = _RunningStat()
        self.metric_any_blocked = _RunningStat()

        if self.shield_folder is not None:
            assert type(self.shield) in [compact_rl.rl.shielding.shields.SelfConstructingShieldOnline, compact_rl.rl.shielding.shields.SelfConstructingShieldOffline], "Saving shield can only be used with self-constructing shields."

    def save_shield(self, path: str, iteration = None):
        """Saves the shield to a file."""
        if not type(self.shield) in [compact_rl.rl.shielding.shields.SelfConstructingShieldOnline, compact_rl.rl.shielding.shields.SelfConstructingShieldOffline]:
            return
        shield_data = ShieldData(
            actions=self.shield.actions,
            original_model_nr_states=self.shield.model_info.model.nr_states,
            observation_to_state=self.shield.model_info.observation_to_state,
            memory=self.shield.memory,
            rounding_precision=self.shield.rounding_precision,
            initial_node=self.shield.initial_node,
            current_action_distributions=self.shield.current_action_distributions
        )
        if iteration is not None:
            path = path + f"-iter-{iteration}-shield.pickle"
        else:
            path = path + f"-shield.pickle"
        with open(path, 'wb') as f:
            pickle.dump(shield_data, f)
        print(f"Shield saved to {path}")

    def load_shield(self, path: str):
        """Loads the shield from a file."""
        with open(path, 'rb') as f:
            shield_data : ShieldData = pickle.load(f)
        assert self.shield.actions == shield_data.actions
        assert self.shield.model_info.model.nr_states == shield_data.original_model_nr_states
        assert self.shield.model_info.observation_to_state == shield_data.observation_to_state
        assert self.shield.memory == shield_data.memory
        assert self.shield.rounding_precision == shield_data.rounding_precision

        self.shield.initial_node = shield_data.initial_node

        self.shield.current_action_distributions = shield_data.current_action_distributions
        if self.shield.memory > 0:
            self.shield.load_matrix_vector_from_current_distributions()
        
        print(f"Shield loaded from {path}")

    def fix_distribution(self, distribution):
        total_prob = sum(distribution)
        if total_prob > 0:
            return [prob / total_prob for prob in distribution]
        else:
            uniform_prob = 1.0 / len(distribution)
            return [uniform_prob for _ in distribution]
        
    def map_played_distribution(self, played_probs, current_state):
        current_state_choice_labels = []

        for choice in range(self.shield.model_info.model.transition_matrix.get_row_group_start(current_state), self.shield.model_info.model.transition_matrix.get_row_group_end(current_state)):
            current_state_choice_labels.append(self.shield.model_info.model.choice_labeling.get_labels_of_choice(choice).pop())

        mapped_played_distribution = [played_probs[self.actions.index(action)] for action in current_state_choice_labels]
        mapped_played_distribution = self.fix_distribution(mapped_played_distribution)

        mapped_played_distribution = [round(prob, self.shield.rounding_precision) for prob in mapped_played_distribution]

        return mapped_played_distribution, current_state_choice_labels

    def _grow_episode_trackers(self, n):
        while len(self._episode_started) < n:
            self._episode_started.append(False)
            self._episode_bad.append(False)
            self._episode_steps.append(0)
            self._episode_blocked.append(0)
            self._episode_first_block_step.append(None)
            self._episode_cost.append(0.0)

    def _finalize_episode(self, i):
        steps = self._episode_steps[i]
        if steps == 0:
            # No actual shield.correct() call happened this "episode" (e.g. immediately
            # terminal) - nothing meaningful to fold into the running statistics.
            return
        blocked = self._episode_blocked[i]
        self.metric_safety.add(1.0 if self._episode_bad[i] else 0.0)
        # allowed_ratio itself is now accumulated per-decision, directly in compute_new_logits -
        # see the note there for why (step-weighted "expected allowed actions", not an unweighted
        # per-episode average).
        self.metric_discounted_cost.add(self._episode_cost[i])
        any_blocked = blocked > 0
        self.metric_any_blocked.add(1.0 if any_blocked else 0.0)
        if any_blocked:
            self.metric_first_block_step.add(self._episode_first_block_step[i])

    def get_episode_stats_summary(self):
        """Returns mean/std/n for the 5 metrics. Every metric but allowed_ratio is one sample per
        finalized episode (i.e. excludes the one dangling, still-in-progress episode per lane at
        the very end of the run, same convention as RiskDistributionShield's own episode-cost
        tracking); allowed_ratio is one sample per individual shield decision instead, so its own
        n (allowed_ratio_n) is a step count, not an episode count - see compute_new_logits."""
        return {
            "episodes_n": self.metric_safety.n,
            "safety_mean": self.metric_safety.mean, "safety_std": self.metric_safety.std,
            "allowed_ratio_n": self.metric_allowed_ratio.n,
            "allowed_ratio_mean": self.metric_allowed_ratio.mean, "allowed_ratio_std": self.metric_allowed_ratio.std,
            "discounted_cost_mean": self.metric_discounted_cost.mean, "discounted_cost_std": self.metric_discounted_cost.std,
            "pct_episodes_blocked_mean": self.metric_any_blocked.mean, "pct_episodes_blocked_std": self.metric_any_blocked.std,
            "first_block_step_n": self.metric_first_block_step.n,
            "first_block_step_mean": self.metric_first_block_step.mean, "first_block_step_std": self.metric_first_block_step.std,
        }

    def compute_new_logits(self, valuations : list, integers : list, prev_actions : list, played_logits : tf.Tensor, resets : list) -> tf.Tensor:
        """ A dummy shielding method that always allows the action.
        Args:
            valuations: The valuations of a current environment state/observation.
            integers: The integer representation of the current environment state/observation.
            prev_actions: The previous actions taken by the agent.
            played_logits: The logits of the actions played by the agent.
            resets: Whether the episode is a restarted simulation in a current state ([True/False]).
        Returns:
            np.ndarray[np.float_]: New logits of probabilities for each action.
        """
        if not self.deterministic_agent:
            played_probs = tf.nn.softmax(played_logits).numpy().tolist()
        else:
            played_probs = tf.one_hot(tf.argmax(played_logits, axis=1), depth=tf.shape(played_logits)[1], dtype=tf.float32).numpy().tolist()
        distributions = []
        self._grow_episode_trackers(len(valuations))

        for i in range(len(valuations)):

            current_state = self.shield.model_info.observation_to_state[integers[i][0]]
            mapped_played_distribution, current_state_choice_labels = self.map_played_distribution(played_probs[i], current_state)

            if resets[i]:
                if self._episode_started[i]:
                    self._finalize_episode(i)
                self._episode_started[i] = True
                self._episode_bad[i] = False
                self._episode_steps[i] = 0
                self._episode_blocked[i] = 0
                self._episode_first_block_step[i] = None
                self._episode_cost[i] = 0.0

            if current_state in self._bad_states_set:
                self._episode_bad[i] = True

            if "goal" in self.shield.model_info.model.labeling.get_labels_of_state(current_state) or "fail" in self.shield.model_info.model.labeling.get_labels_of_state(current_state):
                distribution = mapped_played_distribution
            else:
                if (type(prev_actions[i]) == list):
                    prev_actions_i = prev_actions[i][0]
                else:
                    prev_actions_i = prev_actions[i]

                step_index = self._episode_steps[i]
                blocked_before = self.shield.blocked_actions

                distribution = self.shield.correct(prev_actions_i, current_state, mapped_played_distribution, resets[i], i)

                was_blocked = self.shield.blocked_actions > blocked_before
                # One sample per actual shield decision (not per episode), so the aggregate mean
                # is the step-weighted "expected allowed actions" (matches model-checking's
                # expected_blocked_actions/expected_shield_calls) rather than an unweighted
                # average of each episode's own ratio - the two differ sharply whenever episodes
                # vary a lot in length (a rare episode that gets stuck blocked for 40 steps must
                # count 40x more than a common one that's never blocked in 3 steps, not count the
                # same "one episode" each).
                self.metric_allowed_ratio.add(0.0 if was_blocked else 1.0)
                if was_blocked:
                    self._episode_blocked[i] += 1
                    if self._episode_first_block_step[i] is None:
                        self._episode_first_block_step[i] = step_index
                # L1 distance between what the policy proposed and what the shield actually
                # returned - computed directly here (rather than reading shield.total_intervention_cost,
                # which only RiskDistributionShield tracks) so this is meaningful for every shield type.
                # Identical formula/value to RiskDistributionShield's own internal cost tracking (shields.py),
                # since mapped_played_distribution/distribution here are exactly its correct()'s own
                # distribution/output_distribution.
                cost_delta = sum(abs(p - q) for p, q in zip(mapped_played_distribution, distribution))
                self._episode_cost[i] += (self.discount_factor ** step_index) * cost_delta
                self._episode_steps[i] += 1

            distribution = [distribution[current_state_choice_labels.index(action)] if action in current_state_choice_labels else 0.0 for action in self.actions]

            distributions.append(distribution)

        distributions = np.array(distributions, dtype=np.float_)
        # Convert the boolean mask to logits.
        masked_logits = tf.math.log(distributions + 1e-10)
        return masked_logits

        
