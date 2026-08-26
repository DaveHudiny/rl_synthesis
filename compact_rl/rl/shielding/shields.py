from dataclasses import dataclass
from compact_rl.rl.shielding.model_info import ModelInfo
from compact_rl.rl.shielding.risk_budget import RiskBudgetFunction
import random
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import pickle

import stormpy
import payntbind


def clamp_distribution(distribution, allowed_actions):
    """Clamp a distribution to only allowed actions."""
    total_prob = sum(prob for action, prob in enumerate(distribution) if action in allowed_actions)
    if not total_prob > 0:
        # If no safe actions, return a uniform distribution over allowed actions
        uniform_prob = 1.0 / len(allowed_actions)
        return [uniform_prob if action in allowed_actions else 0.0 for action in range(len(distribution))]
    return [prob / total_prob if action in allowed_actions else 0.0 for action, prob in enumerate(distribution)]

class Shield:
    def __init__(self, model_info: ModelInfo, actions):
        self.model_info = model_info
        self.actions = actions

        # just for evaluation
        self.shield_calls = 0
        self.blocked_actions = 0
        self.added_nonoptimal_actions = 0
        self.trace_count = 0

        self.rounding_precision = 6

    def reset_stats(self):
        self.shield_calls = 0
        self.blocked_actions = 0
        self.added_nonoptimal_actions = 0
        self.trace_count = 0

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        """Correct a distribution."""
        raise NotImplementedError("Not implemented")
    
    def report_info(self):
        return {}

class IdentityShield(Shield):
    def __init__(self, model_info, actions):
        super().__init__(model_info, actions)
    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        return distribution

class StandardShield(Shield):
    def __init__(self, model_info, actions):
        super().__init__(model_info, actions)

    def _allow_action(self, state, action):
        next_val = 0.0
        row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + action
        row = self.model_info.model.transition_matrix.get_row(row_index)
        for entry in row:
            next_val += entry.value() * self.model_info.vmin[entry.column]
        next_val_rounded = round(next_val, self.rounding_precision)
        vmin_state_rounded = round(self.model_info.vmin[state], self.rounding_precision)
        return next_val_rounded <= vmin_state_rounded

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        actions = self.model_info.model.get_nr_available_actions(current_state)
        allowed_actions = [a for a in range(actions) if self._allow_action(current_state, a)]

        # debug only
        for a, prob in enumerate(distribution):
            if prob > 0.0 and a not in allowed_actions:
                self.blocked_actions += 1
                break

        return clamp_distribution(distribution, allowed_actions)

class PessimisticShield(Shield):
    def __init__(self, model_info: ModelInfo, actions, nu: float):
        super().__init__(model_info, actions)
        self.incurred_safeties = [0.0]
        self.path_probs = [1.0]
        self.last_states = [None]
        self.last_distrs = [None]
        self.bmax = self.model_info.vmax[self.model_info.model.initial_states[0]] - nu
        self.standard_shield = StandardShield(model_info, actions)

    def _qmax(self, state, distr):
        qmax = 0.0
        for choice_index, prob in enumerate(distr):
            row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + choice_index
            row = self.model_info.model.transition_matrix.get_row(row_index)
            for entry in row:
                qmax += prob * entry.value() * self.model_info.vmax[entry.column]
        return qmax

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        if trace_index >= len(self.incurred_safeties):
            self.incurred_safeties.append(0.0)
            self.path_probs.append(1.0)
            self.last_states.append(None)
            self.last_distrs.append(None)

        if reset:
            # Reset shield
            self.incurred_safeties[trace_index] = 0.0
            self.path_probs[trace_index] = 1.0
        else:
            assert self.last_states[trace_index] is not None, "Last state is None on non-reset."
            assert self.last_distrs[trace_index] is not None, "Last distribution is None on non-reset."
            last_action_label = self.actions[last_action]

            prev_state_choice_labels = []

            for choice in range(self.model_info.model.transition_matrix.get_row_group_start(self.last_states[trace_index]), self.model_info.model.transition_matrix.get_row_group_end(self.last_states[trace_index])):
                prev_state_choice_labels.append(self.model_info.model.choice_labeling.get_labels_of_choice(choice).pop())

            last_choice_index = prev_state_choice_labels.index(last_action_label)
            row_index = self.model_info.model.transition_matrix.get_row_group_start(self.last_states[trace_index]) + last_choice_index
            row = self.model_info.model.transition_matrix.get_row(row_index)

            last_transition_prob = 0.0
            for entry in row:
                if entry.column == current_state:
                    last_transition_prob = entry.value()
                    break

            self.path_probs[trace_index] = self.path_probs[trace_index] * last_transition_prob * self.last_distrs[trace_index][last_choice_index]


        qmax = self._qmax(current_state, distribution)
        this_step_safety = self.path_probs[trace_index] * (self.model_info.vmax[current_state] - qmax)
        self.incurred_safeties[trace_index] += this_step_safety

        self.last_states[trace_index] = current_state
        self.last_distrs[trace_index] = distribution

        rounded_incurred_safety = round(self.incurred_safeties[trace_index], self.rounding_precision)
        rounded_bmax = round(self.bmax, self.rounding_precision)
        if rounded_incurred_safety >= rounded_bmax:
            output_distribution = distribution
        else:
            new_distribution = self.standard_shield.correct(last_action, current_state, distribution, reset)
            self.incurred_safeties[trace_index] -= this_step_safety
            qmax = self._qmax(current_state, new_distribution)
            this_step_safety = self.path_probs[trace_index] * (self.model_info.vmax[current_state] - qmax)
            self.incurred_safeties[trace_index] += this_step_safety
            self.last_distrs[trace_index] = new_distribution
            output_distribution = new_distribution

            self.blocked_actions = self.standard_shield.blocked_actions

        return output_distribution

class OptimisticShield(Shield):
    def __init__(self, model_info: ModelInfo, actions, nu: float):
        super().__init__(model_info, actions)
        self.incurred_risks = [0.0]
        self.path_probs = [1.0]
        self.last_states = [None]
        self.last_distrs = [None]
        self.bmin = nu - self.model_info.vmin[self.model_info.model.initial_states[0]]
        self.standard_shield = StandardShield(model_info, actions)

    def _qmin(self, state, distr):
        qmin = 0.0
        for choice_index, prob in enumerate(distr):
            row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + choice_index
            row = self.model_info.model.transition_matrix.get_row(row_index)
            for entry in row:
                qmin += prob * entry.value() * self.model_info.vmin[entry.column]
        return qmin

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        if trace_index >= len(self.incurred_risks):
            self.incurred_risks.append(0.0)
            self.path_probs.append(1.0)
            self.last_states.append(None)
            self.last_distrs.append(None)

        if reset:
            # Reset shield
            self.incurred_risks[trace_index] = 0.0
            self.path_probs[trace_index] = 1.0
        else:
            assert self.last_states[trace_index] is not None, "Last state is None on non-reset."
            assert self.last_distrs[trace_index] is not None, "Last distribution is None on non-reset."
            last_action_label = self.actions[last_action]

            prev_state_choice_labels = []

            for choice in range(self.model_info.model.transition_matrix.get_row_group_start(self.last_states[trace_index]), self.model_info.model.transition_matrix.get_row_group_end(self.last_states[trace_index])):
                prev_state_choice_labels.append(self.model_info.model.choice_labeling.get_labels_of_choice(choice).pop())

            last_choice_index = prev_state_choice_labels.index(last_action_label)
            row_index = self.model_info.model.transition_matrix.get_row_group_start(self.last_states[trace_index]) + last_choice_index
            row = self.model_info.model.transition_matrix.get_row(row_index)

            last_transition_prob = 0.0
            for entry in row:
                if entry.column == current_state:
                    last_transition_prob = entry.value()
                    break

            self.path_probs[trace_index] = self.path_probs[trace_index] * last_transition_prob * self.last_distrs[trace_index][last_choice_index]


        qmin = self._qmin(current_state, distribution)
        this_step_risk = self.path_probs[trace_index] * (qmin - self.model_info.vmin[current_state])
        self.incurred_risks[trace_index] += this_step_risk

        self.last_states[trace_index] = current_state
        self.last_distrs[trace_index] = distribution

        rounded_incurred_risk = round(self.incurred_risks[trace_index], self.rounding_precision)
        rounded_bmin = round(self.bmin, self.rounding_precision)
        if rounded_incurred_risk <= rounded_bmin:
            output_distribution = distribution
        else:
            new_distribution = self.standard_shield.correct(last_action, current_state, distribution, reset)
            self.incurred_risks[trace_index] -= this_step_risk
            qmin = self._qmin(current_state, new_distribution)
            this_step_risk = self.path_probs[trace_index] * (qmin - self.model_info.vmin[current_state])
            self.incurred_risks[trace_index] += this_step_risk
            self.last_distrs[trace_index] = new_distribution
            output_distribution = new_distribution

            self.blocked_actions = self.standard_shield.blocked_actions

        return output_distribution
    

class DeltaShield(Shield):
    def __init__(self, model_info: ModelInfo, actions, delta: float):
        super().__init__(model_info, actions)
        self.delta = delta
        self.standard_shield = StandardShield(model_info, actions)

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        # compute expected value of the distribution
        expected_value = 0.0
        for choice_index, prob in enumerate(distribution):
            row_index = self.model_info.model.transition_matrix.get_row_group_start(current_state) + choice_index
            row = self.model_info.model.transition_matrix.get_row(row_index)
            for entry in row:
                expected_value += prob * entry.value() * self.model_info.vmin[entry.column]

        if expected_value - self.model_info.vmin[current_state] < self.delta:
            output_distribution = distribution
        else:
            self.blocked_actions += 1
            output_distribution = self.standard_shield.correct(last_action, current_state, distribution, reset)

        return output_distribution

@dataclass
class Node:
    successors: "dict[tuple[int, int], Node]"
    predecessor: "Node"
    last_played_action: int
    distributions: list[list[float]]
    state_index: int
    value: float
    node_index: int

    def number_of_tree_nodes(self) -> int:
        count = 1
        for succ in self.successors.values():
            count += succ.number_of_tree_nodes()
        return count

    def get_node(self, id) -> "Node":
        if self.node_index == id:
            return self
        for succ in self.successors.values():
            node = succ.get_node(id)
            if node is not None:
                return node
        return None
    
    def get_index_to_node_map(self, index_map={}) -> dict[int, "Node"]:
        index_map[self.node_index] = self
        for succ in self.successors.values():
            succ.get_index_to_node_map(index_map)
        return index_map
    
    def get_state_indices_map(self, index_map={}) -> dict[int, int]:
        index_map[self.node_index] = self.state_index
        for succ in self.successors.values():
            succ.get_state_indices_map(index_map)
        return index_map

    def get_node_to_successor_index_to_state_map(self, node_map={}) -> dict[int, dict[tuple[int, int], int]]:
        succ_map = {}
        for (state, distribution_index), succ in self.successors.items():
            succ_map[(state, distribution_index)] = succ.node_index
        node_map[self.node_index] = succ_map
        for succ in self.successors.values():
            succ.get_node_to_successor_index_to_state_map(node_map)
        return node_map
    

class SelfConstructingShield(Shield):
    def __init__(self, model_info: ModelInfo, actions, nu: float, memory: int = 0):
        super().__init__(model_info, actions)
        self.nu = nu

        self.static_shield_clamp_to_existing_distributions = True # default True, if True then distributions that are not existing in the tree will be clamped to the existing ones allowing for the shield to follow existing nodes for longer but basically blocking the current action

        # assumption
        initial_state = model_info.model.initial_states[0]

        # get vmin
        self.initial_node = Node({}, None, None, [], initial_state, self.model_info.vmin[initial_state], 0)
        self.num_nodes = 1

        self.current_nodes = [self.initial_node]

        self.vmin_actions_distributions = []
        self.vmin_actions = []
        self.initialize_vmin_actions()

        self.last_distribution_indices = [None]

        self.backpropagation_calls = 0
        self.model_checking_calls = 0

        self.current_action_distributions = None

        self.standard_shield = StandardShield(model_info, actions)

        # handle memory limitation
        self.memory = memory
        if self.memory > 0:
            self.safety_property = stormpy.parse_properties("Pmax=? [ F \"bad\" ]")
            self.current_sliding_windows = [[initial_state] + [-1] * (self.memory - 1)]

            if self.memory > 1:
                self.memory_unfolded_model, self.memory_state_to_sliding_window = payntbind.synthesis.createSlidingWindowMemoryMdp(self.model_info.model, self.memory)
            else:
                self.memory_unfolded_model = model_info.model
                self.memory_state_to_sliding_window = [[s] for s in range(self.memory_unfolded_model.nr_states)]
            # Build a reverse lookup dictionary for fast index finding
            self.sliding_window_to_memory_state = {tuple(window): idx for idx, window in enumerate(self.memory_state_to_sliding_window)}

            self.full_transition_matrix_vector = payntbind.synthesis.getVectorFromMatrix(self.memory_unfolded_model.transition_matrix)

            vmin_matrix_vector = []
            self.current_action_distributions = [[] for _ in range(self.memory_unfolded_model.nr_states)]
            for state in range(self.memory_unfolded_model.nr_states):
                original_state = self.memory_state_to_sliding_window[state][0]
                state_vmin_vector = []
                for index, action in enumerate(self.vmin_actions[original_state]):
                    state_vmin_vector.append(self.full_transition_matrix_vector[state][action])
                    self.current_action_distributions[state].append(self.vmin_actions_distributions[original_state][index])
                vmin_matrix_vector.append(state_vmin_vector)

            self.current_matrix_vector = vmin_matrix_vector
            self.current_allow_mdp = payntbind.synthesis.createMdpFromVectorMatrix(self.memory_unfolded_model, self.current_matrix_vector)

    def reset_stats(self):
        super().reset_stats()
        self.backpropagation_calls = 0
        self.model_checking_calls = 0

    def initialize_vmin_actions(self):
        for state in range(self.model_info.model.nr_states):
            actions_count = self.model_info.model.get_nr_available_actions(state)
            vmin_actions_distributions = []
            vmin_actions = []
            for action in range(actions_count):
                row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + action
                row = self.model_info.model.transition_matrix.get_row(row_index)
                val = 0.0
                for entry in row:
                    val += entry.value() * self.model_info.vmin[entry.column]
                val_rounded = round(val, self.rounding_precision)
                vmin_state_rounded = round(self.model_info.vmin[state], self.rounding_precision)
                if val_rounded <= vmin_state_rounded:
                    vmin_actions_distributions.append([1.0 if a == action else 0.0 for a in range(actions_count)])
                    vmin_actions.append(action)
            assert len(vmin_actions) > 0, f"No safe actions for state {state}"
            self.vmin_actions_distributions.append(vmin_actions_distributions)
            self.vmin_actions.append(vmin_actions)

    def load_matrix_vector_from_current_distributions(self, action_distributions=None):
        assert self.memory > 0, "This is only needed for memory bounded shields."

        if action_distributions is None:
            action_distributions = self.current_action_distributions

        for state in range(self.memory_unfolded_model.nr_states):
            original_state = self.memory_state_to_sliding_window[state][0]
            for distr in action_distributions[state]:
                if distr not in self.vmin_actions_distributions[original_state]:
                    row = payntbind.synthesis.createCombinationOfRows(self.full_transition_matrix_vector[state], distr)
                    self.current_matrix_vector[state].append(row)

    def point_in_convex_hull(self, hull_vertices, point, tolerance=1e-6):
        if point in hull_vertices:
            self.convex_point_max_index = hull_vertices.index(point)
            return True
        self.convex_point_max_index = None
        vertices = np.array([tuple(v) for v in hull_vertices])
        point = [val if val >= tolerance else 0.0 for val in point]
        p = np.array(point)
        # Use coordinates directly to check if point is in the convex hull (i.e. the point is linear combination of the basis vectors with non-negative coefficients summing to 1)
        c = np.zeros(len(vertices))
        A_eq = np.vstack([vertices.T, np.ones(len(vertices))])
        b_eq = np.append(p / np.sum(p), 1)
        bounds = [(0, 1)] * len(vertices)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        # print(res.x, res.success, res.status, res.message)
        # Check if the solution coefficients are all within bounds and sum to 1
        if not (res.success and res.status == 0):
            return False
        if np.any(res.x < -tolerance) or np.any(res.x > 1 + tolerance):
            return False
        if not np.isclose(np.sum(res.x), 1, atol=tolerance):
            return False
        if res.x is not None:
            self.convex_point_max_index = np.argmax(res.x)
        else:
            return False
        return True
    
    # this can be optimized probably?
    def back_propagate_values(self, node: Node):
        self.backpropagation_calls += 1
        while node is not None:
            # compute value from successors
            best_value = self.model_info.vmin[node.state_index]
            # points of the convex set
            all_distributions = self.vmin_actions_distributions[node.state_index] + node.distributions
            for distr_index, distr in enumerate(all_distributions):
                q_value = 0.0
                for action_index, action_prob in enumerate(distr):
                    if action_prob == 0:
                        continue
                    row_index = self.model_info.model.transition_matrix.get_row_group_start(node.state_index) + action_index
                    row = self.model_info.model.transition_matrix.get_row(row_index)
                    for entry in row:
                        if (entry.column, distr_index) not in node.successors.keys():
                            q_value += action_prob * entry.value() * self.model_info.vmin[entry.column]
                        else:
                            q_value += action_prob * entry.value() * node.successors[(entry.column, distr_index)].value
                if q_value > best_value:
                    best_value = q_value
            node.value = best_value
            node = node.predecessor

    def correct_for_given_node(self, last_action, current_state, distribution, reset, node):

        assert self.memory == 0, "This is only needed for non-memory bounded shields."

        self.shield_calls += 1

        output_distribution = distribution

        if node is None:
            standard_blocked_actions = self.standard_shield.blocked_actions
            output_distribution = self.standard_shield.correct(last_action, current_state, distribution, reset)
            if self.standard_shield.blocked_actions > standard_blocked_actions:
                self.blocked_actions += 1
            last_distribution_index = None
        else:
            all_allowed_distributions = self.vmin_actions_distributions[current_state] + node.distributions

            if not self.point_in_convex_hull(all_allowed_distributions, distribution):
                self.blocked_actions += 1
                output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])
                if self.static_shield_clamp_to_existing_distributions and output_distribution not in all_allowed_distributions:
                    self.point_in_convex_hull(all_allowed_distributions, output_distribution) # to get the max index
                    output_distribution = all_allowed_distributions[self.convex_point_max_index]
            elif self.static_shield_clamp_to_existing_distributions and output_distribution not in all_allowed_distributions:
                self.blocked_actions += 1
                output_distribution = all_allowed_distributions[self.convex_point_max_index]

            if output_distribution in all_allowed_distributions:
                last_distribution_index = all_allowed_distributions.index(output_distribution)
            else:
                assert not self.static_shield_clamp_to_existing_distributions, "This should not happen, as the distribution was clamped to allowed distributions."
                last_distribution_index = len(all_allowed_distributions)

        return output_distribution, last_distribution_index
    
    def correct_for_given_memory_state(self, last_action, current_state, distribution, reset, memory_state_index):

        assert self.memory > 0, "This is only needed for memory bounded shields."

        self.shield_calls += 1

        output_distribution = distribution

        all_allowed_distributions = self.current_action_distributions[memory_state_index]

        if not self.point_in_convex_hull(all_allowed_distributions, distribution):
            self.blocked_actions += 1
            output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])

        return output_distribution

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        if trace_index >= len(self.current_nodes):
            if self.memory > 0:
                self.current_sliding_windows.append([current_state] + [-1] * (self.memory - 1))
            else:
                self.current_nodes.append(self.initial_node)
                self.last_distribution_indices.append(None)

        if reset:
            self.trace_count += 1
            if self.memory > 0:
                self.current_sliding_windows[trace_index] = [current_state] + [-1] * (self.memory - 1)
                memory_state_index = self.sliding_window_to_memory_state[tuple(self.current_sliding_windows[trace_index])]
            else:
                self.current_nodes[trace_index] = self.initial_node
                self.last_distribution_indices[trace_index] = None
        else:
            if self.memory > 0:
                # update sliding window
                self.current_sliding_windows[trace_index] = [current_state] + self.current_sliding_windows[trace_index][:-1]
                memory_state_index = self.sliding_window_to_memory_state[tuple(self.current_sliding_windows[trace_index])]
            else:
                if self.current_nodes[trace_index] is not None:
                    if (current_state, self.last_distribution_indices[trace_index]) not in self.current_nodes[trace_index].successors.keys():
                        self.current_nodes[trace_index] = None
                    else:
                        self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]

        output_distribution = distribution

        if self.memory > 0:
            if not self.point_in_convex_hull(self.current_action_distributions[memory_state_index], distribution):
                self.blocked_actions += 1
                output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])
        else:
            # check if current distribution is inside the convex set
            if self.current_nodes[trace_index] is None:
                all_allowed_distributions = self.vmin_actions_distributions[current_state]
            else:
                all_allowed_distributions = self.vmin_actions_distributions[current_state] + self.current_nodes[trace_index].distributions
            if not self.point_in_convex_hull(all_allowed_distributions, distribution):
                self.blocked_actions += 1
                output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])
                if self.static_shield_clamp_to_existing_distributions and output_distribution not in all_allowed_distributions:
                    self.point_in_convex_hull(all_allowed_distributions, output_distribution) # to get the max index
                    output_distribution = all_allowed_distributions[self.convex_point_max_index]
            elif self.static_shield_clamp_to_existing_distributions and output_distribution not in all_allowed_distributions:
                self.blocked_actions += 1
                output_distribution = all_allowed_distributions[self.convex_point_max_index]

            if output_distribution in all_allowed_distributions:
                self.last_distribution_indices[trace_index] = all_allowed_distributions.index(output_distribution)
            else:
                assert not self.static_shield_clamp_to_existing_distributions, "This should not happen, as the distribution was clamped to allowed distributions."
                self.last_distribution_indices[trace_index] = len(all_allowed_distributions)

        return output_distribution

class SelfConstructingShieldOffline(SelfConstructingShield):
    def __init__(self, model_info: ModelInfo, actions, nu: float, memory: int = 0):
        super().__init__(model_info, actions, nu, memory=memory)

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        if trace_index >= len(self.current_nodes):
            if self.memory > 0:
                self.current_sliding_windows.append([current_state] + [-1] * (self.memory - 1))
            else:
                self.current_nodes.append(self.initial_node)
                self.last_distribution_indices.append(None)

        if reset:
            self.trace_count += 1
            # we are looking at a different trace now, so we need to update the values on the previous trace
            if self.memory > 0:
                self.current_sliding_windows[trace_index] = [current_state] + [-1] * (self.memory - 1)
                memory_state_index = self.sliding_window_to_memory_state[tuple(self.current_sliding_windows[trace_index])]
            else:
                if len(self.initial_node.successors) > 0:
                    self.back_propagate_values(self.current_nodes[trace_index]) # TODO THIS (maybe???) DOES NOT WORK PROPERLY IF MULTIPLE TRACES ARE USED SIMULTANEOUSLY
                self.current_nodes[trace_index] = self.initial_node
                self.last_distribution_indices[trace_index] = None
        else:
            if self.memory > 0:
                # update sliding window
                self.current_sliding_windows[trace_index] = [current_state] + self.current_sliding_windows[trace_index][:-1]
                memory_state_index = self.sliding_window_to_memory_state[tuple(self.current_sliding_windows[trace_index])]
            else:
                if (current_state, self.last_distribution_indices[trace_index]) not in self.current_nodes[trace_index].successors.keys():
                    all_dist = self.vmin_actions_distributions[self.current_nodes[trace_index].state_index] + self.current_nodes[trace_index].distributions
                    assert self.last_distribution_indices[trace_index] is None and len(all_dist) == 0 or self.last_distribution_indices[trace_index] < len(all_dist)
                    self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])] = Node({}, self.current_nodes[trace_index], self.last_distribution_indices[trace_index], [], current_state, self.model_info.vmin[current_state], self.num_nodes)
                    self.num_nodes += 1
                    self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]
                else:
                    self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]

        output_distribution = distribution

        if self.memory > 0:
            if not self.point_in_convex_hull(self.current_action_distributions[memory_state_index], distribution):
                weighted_row = payntbind.synthesis.createCombinationOfRows(self.full_transition_matrix_vector[memory_state_index], distribution)
                self.current_matrix_vector[memory_state_index].append(weighted_row)
                self.current_allow_mdp = payntbind.synthesis.createMdpFromVectorMatrix(self.memory_unfolded_model, self.current_matrix_vector)
                result = stormpy.model_checking(self.current_allow_mdp, self.safety_property[0])
                self.model_checking_calls += 1
                if result.at(self.current_allow_mdp.initial_states[0]) > self.nu:
                    self.blocked_actions += 1
                    self.current_matrix_vector[memory_state_index].pop()
                    output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])
                else:
                    self.current_action_distributions[memory_state_index].append(distribution)
                    self.added_nonoptimal_actions += 1
        else:
            # check if current distribution is inside the convex set
            if not self.point_in_convex_hull(self.current_nodes[trace_index].distributions + self.vmin_actions_distributions[current_state], distribution):
                self.current_nodes[trace_index].distributions.append(distribution)

                self.back_propagate_values(self.current_nodes[trace_index])

                if self.initial_node.value > self.nu:
                    self.blocked_actions += 1
                    self.current_nodes[trace_index].distributions.pop()
                    self.back_propagate_values(self.current_nodes[trace_index])
                    output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])
                else:
                    self.added_nonoptimal_actions += 1

            # Change compared to parent class: last_played_action now stores the index of the played distribution
            if output_distribution in self.vmin_actions_distributions[current_state]:
                self.last_distribution_indices[trace_index] = self.vmin_actions_distributions[current_state].index(output_distribution)
            elif output_distribution in self.current_nodes[trace_index].distributions:
                self.last_distribution_indices[trace_index] = len(self.vmin_actions_distributions[current_state]) + self.current_nodes[trace_index].distributions.index(output_distribution)
            else:                
                self.current_nodes[trace_index].distributions.append(output_distribution)
                self.last_distribution_indices[trace_index] = len(self.vmin_actions_distributions[current_state]) + len(self.current_nodes[trace_index].distributions) - 1

        return output_distribution


class SelfConstructingShieldOnline(SelfConstructingShield):
    def __init__(self, model_info: ModelInfo, actions, nu: float, memory: int = 0):
        super().__init__(model_info, actions, nu, memory=memory)
        self.blocked_distributions = [[]]

    def finalize_all_unfinished_traces(self):
        for trace_index in range(len(self.blocked_distributions)):
            self.explore_blocked_distributions(self.blocked_distributions[trace_index])
            self.blocked_distributions[trace_index] = []
    
    def explore_blocked_distributions(self, blocked_distributions):
        if self.memory > 0:
            for memory_state_index, distribution in blocked_distributions:
                if self.point_in_convex_hull(self.current_action_distributions[memory_state_index], distribution):
                    continue
                weighted_row = payntbind.synthesis.createCombinationOfRows(self.full_transition_matrix_vector[memory_state_index], distribution)
                self.current_matrix_vector[memory_state_index].append(weighted_row)
                self.current_allow_mdp = payntbind.synthesis.createMdpFromVectorMatrix(self.memory_unfolded_model, self.current_matrix_vector)
                result = stormpy.model_checking(self.current_allow_mdp, self.safety_property[0])
                self.model_checking_calls += 1
                if result.at(self.current_allow_mdp.initial_states[0]) > self.nu:
                    self.current_matrix_vector[memory_state_index].pop()
                else:
                    self.current_action_distributions[memory_state_index].append(distribution)
                    self.added_nonoptimal_actions += 1
        else:
            value_restore_needed = False
            for node, distribution in blocked_distributions:
                if self.point_in_convex_hull(node.distributions + self.vmin_actions_distributions[node.state_index], distribution):
                    continue
                node.distributions.append(distribution)
                self.back_propagate_values(node)
                if self.initial_node.value > self.nu:
                    node.distributions.pop()
                    value_restore_needed = True
                else:
                    self.added_nonoptimal_actions += 1
                    value_restore_needed = False

            # if for the last node the action was not allowed, we need to perform one more back-propagation to have the correct values in the tree
            if value_restore_needed:
                self.back_propagate_values(blocked_distributions[-1][0])

    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):
        self.shield_calls += 1

        if trace_index >= len(self.current_nodes):
            if self.memory > 0:
                self.current_sliding_windows.append([current_state] + [-1] * (self.memory - 1))
            else:
                self.current_nodes.append(self.initial_node)
                self.last_distribution_indices.append(None)
            self.blocked_distributions.append([])

        if reset:
            self.trace_count += 1
            # we are looking at a different trace now, so we are going to update the shield we are using
            self.explore_blocked_distributions(self.blocked_distributions[trace_index])
            self.blocked_distributions[trace_index] = []

            if self.memory > 0:
                self.current_sliding_windows[trace_index] = [current_state] + [-1] * (self.memory - 1)
                memory_state_index = self.sliding_window_to_memory_state[tuple(self.current_sliding_windows[trace_index])]
            else:
                self.current_nodes[trace_index] = self.initial_node
                self.last_distribution_indices[trace_index] = None
        else:
            if self.memory > 0:
                # update sliding window
                self.current_sliding_windows[trace_index] = [current_state] + self.current_sliding_windows[trace_index][:-1]
                memory_state_index = self.sliding_window_to_memory_state[tuple(self.current_sliding_windows[trace_index])]
            else:
                # Change compared to parent class: last_played_action now stores the index of the played distribution
                if (current_state, self.last_distribution_indices[trace_index]) not in self.current_nodes[trace_index].successors.keys():
                    all_dist = self.vmin_actions_distributions[self.current_nodes[trace_index].state_index] + self.current_nodes[trace_index].distributions
                    assert self.last_distribution_indices[trace_index] is None and len(all_dist) == 0 or self.last_distribution_indices[trace_index] < len(all_dist)
                    self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])] = Node({}, self.current_nodes[trace_index], self.last_distribution_indices[trace_index], [], current_state, self.model_info.vmin[current_state], self.num_nodes)
                    self.num_nodes += 1
                    self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]
                else:
                    self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]

        output_distribution = distribution

        # check if current distribution is inside the convex set
        if self.memory > 0:
            if not self.point_in_convex_hull(self.current_action_distributions[memory_state_index], distribution):
                self.blocked_actions += 1
                self.blocked_distributions[trace_index].append((memory_state_index, distribution))
                output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])
        else:
            if not self.point_in_convex_hull(self.current_nodes[trace_index].distributions + self.vmin_actions_distributions[current_state], distribution):
                self.blocked_actions += 1
                self.blocked_distributions[trace_index].append((self.current_nodes[trace_index], distribution))
                output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])

            # Change compared to parent class: last_played_action now stores the index of the played distribution
            if output_distribution in self.vmin_actions_distributions[current_state]:
                self.last_distribution_indices[trace_index] = self.vmin_actions_distributions[current_state].index(output_distribution)
            elif output_distribution in self.current_nodes[trace_index].distributions:
                self.last_distribution_indices[trace_index] = len(self.vmin_actions_distributions[current_state]) + self.current_nodes[trace_index].distributions.index(output_distribution)
            else:
                self.current_nodes[trace_index].distributions.append(output_distribution)
                self.last_distribution_indices[trace_index] = len(self.vmin_actions_distributions[current_state]) + len(self.current_nodes[trace_index].distributions) - 1

        return output_distribution


class ShieldWithBudget(Shield):

    def __init__(self, model_info: ModelInfo, actions, nu: float, budget: RiskBudgetFunction):
        super().__init__(model_info, actions)
        self.nu = nu
        self.budget = budget
        self.vmin_actions = []

        self._compute_vmin_actions()

        # assumption
        initial_state = model_info.model.initial_states[0]

        self.vmax_at_initial_state = self.model_info.vmax[initial_state]

        self.remaining_risk = [min(self.nu, self.vmax_at_initial_state)]
        self.history = [[]]
        self.last_states = [None]
        self.last_distributions = [None]
        self.last_qmin_ds = [None]


    def _compute_vmin_actions(self):
        for state in range(self.model_info.model.nr_states):
            actions_count = self.model_info.model.get_nr_available_actions(state)
            vmin_actions = []
            for action in range(actions_count):
                row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + action
                row = self.model_info.model.transition_matrix.get_row(row_index)
                val = 0.0
                for entry in row:
                    val += entry.value() * self.model_info.vmin[entry.column]
                val_rounded = round(val, self.rounding_precision)
                vmin_state_rounded = round(self.model_info.vmin[state], self.rounding_precision)
                if val_rounded <= vmin_state_rounded:
                    vmin_actions.append(action)
            assert len(vmin_actions) > 0, f"No safe actions for state {state}"
            self.vmin_actions.append(vmin_actions)


    def _qmin(self, state, distr):
        qmin = 0.0
        for choice_index, prob in enumerate(distr):
            row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + choice_index
            row = self.model_info.model.transition_matrix.get_row(row_index)
            for entry in row:
                qmin += prob * entry.value() * self.model_info.vmin[entry.column]
        return qmin

    def _qmax(self, state, distr):
        qmax = 0.0
        for choice_index, prob in enumerate(distr):
            row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + choice_index
            row = self.model_info.model.transition_matrix.get_row(row_index)
            for entry in row:
                qmax += prob * entry.value() * self.model_info.vmax[entry.column]
        return qmax

    def _transition_prob(self, last_state, distribution, action, current_state):
        row_index = self.model_info.model.transition_matrix.get_row_group_start(last_state) + action
        row = self.model_info.model.transition_matrix.get_row(row_index)
        prob = 0.0
        for entry in row:
            if entry.column == current_state:
                prob += distribution[action] * entry.value()
        return prob

    def _local_action_index(self, state, global_action):
        # `global_action` indexes into self.actions (as passed to Shield.correct), but rows in
        # the transition matrix are indexed per-state by local choice index - translate via the
        # shared action label, same as PessimisticShield/OptimisticShield.correct do inline.
        action_label = self.actions[global_action]
        row_group_start = self.model_info.model.transition_matrix.get_row_group_start(state)
        row_group_end = self.model_info.model.transition_matrix.get_row_group_end(state)
        choice_labels = [self.model_info.model.choice_labeling.get_labels_of_choice(choice).pop() for choice in range(row_group_start, row_group_end)]
        return choice_labels.index(action_label)


    def correct(self, last_action : int, current_state : int, distribution : list[float], reset : bool, trace_index : int = 0):

        self.shield_calls += 1

        if trace_index >= len(self.remaining_risk):
            self.remaining_risk.append(min(self.nu, self.vmax_at_initial_state))
            self.history.append([])
            self.last_states.append(None)
            self.last_distributions.append(None)
            self.last_qmin_ds.append(None)

        if reset:
            self.remaining_risk[trace_index] = min(self.nu, self.vmax_at_initial_state)
            self.history[trace_index] = []
        else:
            assert self.last_states[trace_index] is not None, "Last state is None on non-reset."
            assert self.last_distributions[trace_index] is not None, "Last distribution is None on non-reset."

            local_last_action = self._local_action_index(self.last_states[trace_index], last_action)

            self.history[trace_index].append(local_last_action)

            transition_prob = self._transition_prob(self.last_states[trace_index], self.last_distributions[trace_index], local_last_action, current_state)
            # this shouldn't happen so I will just put assert here, but for completness of the risk function if this happens it should either
            # be set to Vmax(current_state) if qmin <= risk or to Vmin(current_state) otherwise
            assert transition_prob > 0, f"Transition probability is zero for last_state {self.last_states[trace_index]}, last_action {local_last_action}, current_state {current_state}."

            risk_budget_distribution = self.budget(self.history[trace_index], self.last_distributions[trace_index])
            risk_budget = risk_budget_distribution.get((local_last_action, current_state), 0.0)
            qmax = self._qmax(self.last_states[trace_index], self.last_distributions[trace_index])
            slack = min(self.remaining_risk[trace_index], qmax) - self.last_qmin_ds[trace_index]
            self.remaining_risk[trace_index] = self.model_info.vmin[current_state] + (risk_budget*slack)/transition_prob


        self.last_qmin_ds[trace_index] = self._qmin(current_state, distribution)

        if self.last_qmin_ds[trace_index] > self.remaining_risk[trace_index]:
            self.blocked_actions += 1
            output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])
            self.last_qmin_ds[trace_index] = self._qmin(current_state, output_distribution) # this needs to be updated because the distribution was changed, otherwise the remaining risk will be wrong in the next step
        else:
            output_distribution = distribution


        self.last_states[trace_index] = current_state
        self.last_distributions[trace_index] = output_distribution
        self.history[trace_index].append(current_state)
        self.history[trace_index].append(output_distribution)

        return output_distribution