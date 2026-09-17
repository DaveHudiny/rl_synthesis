from compact_rl.rl.tools.args_emulator import ArgsEmulator
import tensorflow as tf

from compact_rl.rl.shielding.model_info import ModelInfo
import compact_rl.rl.shielding.shields
from compact_rl.rl.shielding.constructed_shield_data import ShieldData
from compact_rl.rl.shielding.risk_budget import BUDGET_FUNCTIONS, NNRiskBudget

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
    def __init__(self, actions : list[str], model : stormpy.storage.SparsePomdp, nu : float, shield_type : str, args : ArgsEmulator = None, shield_memory : int = 0, debug: bool = False, shield_folder: str = None, deterministic_agent: bool = False, budget: str = "uniform", use_clamp: bool = False, environment=None, budget_checkpoint: str = None, force_wasteless_budget: bool = True, discount_factor: float = 0.99):
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
            reward_formula = stormpy.parse_properties("Rmax=? [ C<=50 ]")
            reward_result = stormpy.model_checking(mdp, reward_formula[0])
            print("Max expected rewards to goal from initial state:", reward_result.get_values()[mdp.initial_states[0]])
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
        elif shield_type == 'budget':
            if budget == 'nn':
                assert environment is not None, "budget='nn' requires ShieldProcessor's environment argument (for state features)."
                assert budget_checkpoint is not None, "budget='nn' requires --budget-checkpoint."
                budget_fn = self._load_nn_budget(model_info, environment, nu, budget_checkpoint)
            elif budget == 'nn-reinforce':
                assert environment is not None, "budget='nn-reinforce' requires ShieldProcessor's environment argument (for state features)."
                assert budget_checkpoint is not None, "budget='nn-reinforce' requires --budget-checkpoint."
                budget_fn = self._load_nn_reinforce_budget(model_info, environment, nu, budget_checkpoint)
            else:
                budget_fn = BUDGET_FUNCTIONS[budget](model_info)
            self.shield = compact_rl.rl.shielding.shields.ShieldWithBudget(model_info=model_info, actions=self.actions, nu=nu, budget=budget_fn, use_l1_projection=not use_clamp, force_wasteless_budget=force_wasteless_budget, discount_factor=discount_factor)
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

    def _load_nn_budget(self, model_info, environment, nu, budget_checkpoint):
        """Builds a RiskBudgetActorNetwork matching train_risk_budget_shield.py's own
        conventions, restores its weights from a checkpoint saved by that script, and wraps it
        in NNRiskBudget for use as this shield's budget function. The full ppo_agent.PPOAgent
        (not just the actor net) has to be reconstructed to match the object graph the
        checkpoint was actually saved with (tf.train.Checkpoint(agent=...) in that script);
        only the restored actor net is kept afterward. `policy=None` is safe here: the
        RiskBudgetTrainingEnv instance is only used for its (policy-independent) specs, never
        stepped."""
        from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
        from compact_rl.rl.shielding.risk_budget_training_env import RiskBudgetTrainingEnv
        from compact_rl.rl.shielding.train_risk_budget_shield import build_agent, load_agent

        train_env = RiskBudgetTrainingEnv(environment=environment, policy=None, model_info=model_info,
                                           actions=self.actions, nu=nu)
        tf_train_env = TFPyEnvironment(train_env)
        agent = build_agent(train_env, tf_train_env)
        load_agent(agent, budget_checkpoint)
        return NNRiskBudget(model_info, agent._actor_net, self.actions, environment.observation_valuations)

    def _load_nn_reinforce_budget(self, model_info, environment, nu, budget_checkpoint):
        """Same idea as _load_nn_budget, but for a checkpoint saved by
        train_risk_budget_reinforce.py, which checkpoints only the bare actor net
        (tf.train.Checkpoint(actor_net=actor_net), no PPOAgent/value_net involved) - so no
        agent reconstruction is needed here, just a freshly-built, correctly-specced actor
        net to restore weights into."""
        from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
        from compact_rl.rl.shielding.risk_budget_training_env import RiskBudgetTrainingEnv
        from compact_rl.rl.shielding.train_risk_budget_reinforce import build_actor, load_actor

        train_env = RiskBudgetTrainingEnv(environment=environment, policy=None, model_info=model_info,
                                           actions=self.actions, nu=nu)
        tf_train_env = TFPyEnvironment(train_env)
        actor_net, _ = build_actor(train_env, tf_train_env)
        load_actor(actor_net, budget_checkpoint)
        return NNRiskBudget(model_info, actor_net, self.actions, environment.observation_valuations)

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

    def save_budget(self, path: str):
        """Saves the shield's budget function's learned state to a file, if it supports it."""
        if not hasattr(self.shield, "budget") or not hasattr(self.shield.budget, "save_counts"):
            return
        self.shield.budget.save_counts(path)
        print(f"Budget saved to {path}")

    def load_budget(self, path: str):
        """Loads the shield's budget function's learned state from a file, if it supports it."""
        assert hasattr(self.shield, "budget") and hasattr(self.shield.budget, "load_counts"), "Loading a budget can only be used with shields whose budget function supports it."
        self.shield.budget.load_counts(path)
        print(f"Budget loaded from {path}")

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
        self.metric_allowed_ratio.add(1.0 - blocked / steps)
        self.metric_discounted_cost.add(self._episode_cost[i])
        any_blocked = blocked > 0
        self.metric_any_blocked.add(1.0 if any_blocked else 0.0)
        if any_blocked:
            self.metric_first_block_step.add(self._episode_first_block_step[i])

    def get_episode_stats_summary(self):
        """Returns mean/std/n for the 5 per-episode metrics, over every episode that was
        actually finalized (i.e. excludes the one dangling, still-in-progress episode per lane
        at the very end of the run, same convention as ShieldWithBudget's own episode-cost
        tracking)."""
        return {
            "episodes_n": self.metric_safety.n,
            "safety_mean": self.metric_safety.mean, "safety_std": self.metric_safety.std,
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
                cost_before = getattr(self.shield, "total_intervention_cost", 0.0)

                distribution = self.shield.correct(prev_actions_i, current_state, mapped_played_distribution, resets[i], i)

                if self.shield.blocked_actions > blocked_before:
                    self._episode_blocked[i] += 1
                    if self._episode_first_block_step[i] is None:
                        self._episode_first_block_step[i] = step_index
                cost_delta = getattr(self.shield, "total_intervention_cost", 0.0) - cost_before
                self._episode_cost[i] += (self.discount_factor ** step_index) * cost_delta
                self._episode_steps[i] += 1

            distribution = [distribution[current_state_choice_labels.index(action)] if action in current_state_choice_labels else 0.0 for action in self.actions]

            distributions.append(distribution)

        distributions = np.array(distributions, dtype=np.float_)
        # Convert the boolean mask to logits.
        masked_logits = tf.math.log(distributions + 1e-10)
        return masked_logits

        
