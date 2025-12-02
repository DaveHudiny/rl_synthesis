from dataclasses import dataclass
from shielding.model_info import ModelInfo
import random
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

import stormpy


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

    def correct(self, last_action: any, current_state: any, distribution, reset, trace_index=0):
        """Correct a distribution."""
        raise NotImplementedError("Not implemented")
    
    def report_info(self):
        return {}

class IdentityShield(Shield):
    def __init__(self, model_info, actions):
        super().__init__(model_info, actions)
    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
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
        return next_val <= self.model_info.vmin[state]

    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
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

    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
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
            last_action = last_action[0]
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

        if self.incurred_safeties[trace_index] >= self.bmax:
            output_distribution = distribution
        else:
            self.blocked_actions += 1

            new_distribution = self.standard_shield.correct(last_action, current_state, distribution, reset)
            self.incurred_safeties[trace_index] -= this_step_safety
            qmax = self._qmax(current_state, new_distribution)
            this_step_safety = self.path_probs[trace_index] * (self.model_info.vmax[current_state] - qmax)
            self.incurred_safeties[trace_index] += this_step_safety
            self.last_distrs[trace_index] = new_distribution
            output_distribution = new_distribution

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

    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
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
            last_action = last_action[0]
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

        if self.incurred_risks[trace_index] <= self.bmin:
            output_distribution = distribution
        else:
            self.blocked_actions += 1

            new_distribution = self.standard_shield.correct(last_action, current_state, distribution, reset)
            self.incurred_risks[trace_index] -= this_step_risk
            qmin = self._qmin(current_state, new_distribution)
            this_step_risk = self.path_probs[trace_index] * (qmin - self.model_info.vmin[current_state])
            self.incurred_risks[trace_index] += this_step_risk
            self.last_distrs[trace_index] = new_distribution
            output_distribution = new_distribution

        return output_distribution
    

class DeltaShield(Shield):
    def __init__(self, model_info: ModelInfo, actions, delta: float):
        super().__init__(model_info, actions)
        self.delta = delta
        self.standard_shield = StandardShield(model_info, actions)

    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
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

    def number_of_tree_nodes(self) -> int:
        count = 1
        for succ in self.successors.values():
            count += succ.number_of_tree_nodes()
        return count

class SelfConstructingShieldDeterministicUnsafe(Shield):

    def __init__(self, model_info: ModelInfo, actions, nu: float, memory: int = 0):
        super().__init__(model_info, actions)
        self.nu = nu

        # assumption
        initial_state = model_info.model.initial_states[0]

        # get vmin
        self.initial_node = Node({}, None, None, [], initial_state, self.model_info.vmin[initial_state])

        self.current_nodes = [self.initial_node]

        self.vmin_actions_distributions = []
        self.vmin_actions = []
        self.initialize_vmin_actions()

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
                if val <= self.model_info.vmin[state]:
                    vmin_actions_distributions.append([1.0 if a == action else 0.0 for a in range(actions_count)])
                    vmin_actions.append(action)
            assert len(vmin_actions) > 0, f"No safe actions for state {state}"
            self.vmin_actions_distributions.append(vmin_actions_distributions)
            self.vmin_actions.append(vmin_actions)

    # this can be optimized probably?
    def back_propagate_values(self, node: Node):
        while node is not None:
            # compute value from successors
            best_value = float('-inf')
            # points of the convex set
            all_distributions = node.distributions + self.vmin_actions_distributions[node.state_index]
            for distr in all_distributions:
                q_value = 0.0
                for action_index, action_prob in enumerate(distr):
                    if action_prob == 0:
                        continue
                    row_index = self.model_info.model.transition_matrix.get_row_group_start(node.state_index) + action_index
                    row = self.model_info.model.transition_matrix.get_row(row_index)
                    for entry in row:
                        if (entry.column, action_index) not in node.successors.keys():
                            q_value += action_prob * entry.value() * self.model_info.vmin[entry.column]
                        else:
                            q_value += action_prob * entry.value() * node.successors[(entry.column, action_index)].value
                if q_value > best_value:
                    best_value = q_value
            assert best_value != float('-inf')
            # print(best_value)
            node.value = best_value
            node = node.predecessor

    def point_in_convex_hull(self, hull_vertices, point, tolerance=1e-12):
        vertices = np.array([tuple(v) for v in hull_vertices])
        p = np.array(point)
        # Use coordinates directly to check if point is in the convex hull (i.e. the point is linear combination of the basis vectors with non-negative coefficients summing to 1)
        c = np.zeros(len(vertices))
        A_eq = np.vstack([vertices.T, np.ones(len(vertices))])
        b_eq = np.append(p, 1)
        bounds = [(0, 1)] * len(vertices)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        # Check if the solution coefficients are all within bounds and sum to 1
        if not (res.success and res.status == 0):
            return False
        if np.any(res.x < -tolerance) or np.any(res.x > 1 + tolerance):
            return False
        if not np.isclose(np.sum(res.x), 1, atol=tolerance):
            return False
        return True

    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
        self.shield_calls += 1

        if trace_index >= len(self.current_nodes):
            self.current_nodes.append(self.initial_node)

        if reset:
            self.trace_count += 1
            # we are looking at a different trace now, so we need to update the values on the previous trace
            if len(self.initial_node.successors) > 0:
                self.back_propagate_values(self.current_nodes[trace_index]) # TODO THIS DOES NOT WORK PROPERLY IF MULTIPLE TRACES ARE USED SIMULTANEOUSLY
            self.current_nodes[trace_index] = self.initial_node
            last_action = None
        else:
            last_action = last_action[0]
            last_action_label = self.actions[last_action]

            prev_state_choice_labels = []
            for choice in range(self.model_info.model.transition_matrix.get_row_group_start(self.current_nodes[trace_index].state_index), self.model_info.model.transition_matrix.get_row_group_end(self.current_nodes[trace_index].state_index)):
                prev_state_choice_labels.append(self.model_info.model.choice_labeling.get_labels_of_choice(choice).pop())

            last_choice_index = prev_state_choice_labels.index(last_action_label)
            last_action = last_choice_index

            if (current_state, last_action) not in self.current_nodes[trace_index].successors.keys():
                self.current_nodes[trace_index].successors[(current_state, last_action)] = Node({}, self.current_nodes[trace_index], last_action, [], current_state, self.model_info.vmin[current_state])
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, last_action)]
            else:
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, last_action)]

        output_distribution = distribution

        # check if current distribution is inside the convex set
        if not self.point_in_convex_hull(self.current_nodes[trace_index].distributions + self.vmin_actions_distributions[current_state], distribution):
            self.current_nodes[trace_index].distributions.append(distribution)

            self.back_propagate_values(self.current_nodes[trace_index])

            if self.initial_node.value > self.nu:
                self.blocked_actions += 1
                self.current_nodes[trace_index].distributions.pop()
                output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])

        return output_distribution
    

# # TODO think about nice implementation where you could easily parameterize the behaviour anywhere between the two version of the self-constructing shield
class SelfConstructingShieldUnsafe(SelfConstructingShieldDeterministicUnsafe):
    def __init__(self, model_info: ModelInfo, actions, nu: float, memory: int = 0):
        super().__init__(model_info, actions, nu, memory)

        self.last_distribution_indices = [None]

    # this can be optimized probably?
    def back_propagate_values(self, node: Node):
        while node is not None:
            # compute value from successors
            best_value = float('-inf')
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
            assert best_value != float('-inf')
            # print(best_value)
            node.value = best_value
            node = node.predecessor

    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
        self.shield_calls += 1

        if trace_index >= len(self.current_nodes):
            self.current_nodes.append(self.initial_node)
            self.last_distribution_indices.append(None)

        if reset:
            self.trace_count += 1
            # we are looking at a different trace now, so we need to update the values on the previous trace
            if len(self.initial_node.successors) > 0:
                self.back_propagate_values(self.current_nodes[trace_index]) # TODO THIS DOES NOT WORK PROPERLY IF MULTIPLE TRACES ARE USED SIMULTANEOUSLY
            self.current_nodes[trace_index] = self.initial_node
            self.last_distribution_indices[trace_index] = None
        else:
            # Change compared to parent class: last_played_action now stores the index of the played distribution
            if (current_state, self.last_distribution_indices[trace_index]) not in self.current_nodes[trace_index].successors.keys():
                all_dist = self.vmin_actions_distributions[self.current_nodes[trace_index].state_index] + self.current_nodes[trace_index].distributions
                assert self.last_distribution_indices[trace_index] is None and len(all_dist) == 0 or self.last_distribution_indices[trace_index] < len(all_dist)
                self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])] = Node({}, self.current_nodes[trace_index], self.last_distribution_indices[trace_index], [], current_state, self.model_info.vmin[current_state])
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]
            else:
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]

        output_distribution = distribution

        # check if current distribution is inside the convex set
        if not self.point_in_convex_hull(self.current_nodes[trace_index].distributions + self.vmin_actions_distributions[current_state], distribution):
            self.current_nodes[trace_index].distributions.append(distribution)

            self.back_propagate_values(self.current_nodes[trace_index])

            if self.initial_node.value > self.nu:
                self.blocked_actions += 1
                self.current_nodes[trace_index].distributions.pop()
                output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])

        # Change compared to parent class: last_played_action now stores the index of the played distribution
        if output_distribution in self.vmin_actions_distributions[current_state]:
            self.last_distribution_indices[trace_index] = self.vmin_actions_distributions[current_state].index(output_distribution)
        elif output_distribution in self.current_nodes[trace_index].distributions:
            self.last_distribution_indices[trace_index] = len(self.vmin_actions_distributions[current_state]) + self.current_nodes[trace_index].distributions.index(output_distribution)
        else:
            self.added_nonoptimal_actions += 1
            self.current_nodes[trace_index].distributions.append(output_distribution)
            self.last_distribution_indices[trace_index] = len(self.vmin_actions_distributions[current_state]) + len(self.current_nodes[trace_index].distributions) - 1

        return output_distribution



class SelfConstructingShieldDeterministic(Shield):

    def __init__(self, model_info: ModelInfo, actions, nu: float, memory: int = 0):
        super().__init__(model_info, actions)
        self.nu = nu

        # assumption
        initial_state = model_info.model.initial_states[0]

        # get vmin
        self.initial_node = Node({}, None, None, [], initial_state, self.model_info.vmin[initial_state])

        self.current_nodes = [self.initial_node]
        self.blocked_distributions = [[]]

        self.vmin_actions_distributions = []
        self.vmin_actions = []
        self.initialize_vmin_actions()

        # handle memory limitation
        self.memory = memory
        if self.memory > 0:
            self.safety_property = stormpy.parse_properties("Pmax=? [ F \"bad\" ]")

            assert self.memory == 1, "Currently only memory<=1 limitation is supported."
            self.memory_unfolded_model = model_info.model # TODO implement memory unfolding
            self.memory_state_to_original_mapping = [s for s in range(self.memory_unfolded_model.nr_states)] # TODO implement memory unfolding

            self.allowed_choices = []
            for state in range(self.memory_unfolded_model.nr_states):
                original_state = self.memory_state_to_original_mapping[state]
                row_start = self.memory_unfolded_model.transition_matrix.get_row_group_start(state)

                for vmin_action in self.vmin_actions[original_state]:
                    choice_index = row_start + vmin_action
                    self.allowed_choices.append(choice_index)
                

    # TODO careful here with the float comparisons!
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
                if val <= self.model_info.vmin[state]:
                    vmin_actions_distributions.append([1.0 if a == action else 0.0 for a in range(actions_count)])
                    vmin_actions.append(action)
            assert len(vmin_actions) > 0, f"No safe actions for state {state}"
            self.vmin_actions_distributions.append(vmin_actions_distributions)
            self.vmin_actions.append(vmin_actions)

    # this can be optimized probably?
    def back_propagate_values(self, node: Node):
        while node is not None:
            # compute value from successors
            best_value = float('-inf')
            # points of the convex set
            all_distributions = node.distributions + self.vmin_actions_distributions[node.state_index]
            for distr in all_distributions:
                q_value = 0.0
                for action_index, action_prob in enumerate(distr):
                    if action_prob == 0:
                        continue
                    row_index = self.model_info.model.transition_matrix.get_row_group_start(node.state_index) + action_index
                    row = self.model_info.model.transition_matrix.get_row(row_index)
                    for entry in row:
                        if (entry.column, action_index) not in node.successors.keys():
                            q_value += action_prob * entry.value() * self.model_info.vmin[entry.column]
                        else:
                            q_value += action_prob * entry.value() * node.successors[(entry.column, action_index)].value
                if q_value > best_value:
                    best_value = q_value
            assert best_value != float('-inf')
            # print(best_value)
            node.value = best_value
            node = node.predecessor

    def point_in_convex_hull(self, hull_vertices, point, tolerance=1e-12):
        vertices = np.array([tuple(v) for v in hull_vertices])
        p = np.array(point)
        # Use coordinates directly to check if point is in the convex hull (i.e. the point is linear combination of the basis vectors with non-negative coefficients summing to 1)
        c = np.zeros(len(vertices))
        A_eq = np.vstack([vertices.T, np.ones(len(vertices))])
        b_eq = np.append(p, 1)
        bounds = [(0, 1)] * len(vertices)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        # Check if the solution coefficients are all within bounds and sum to 1
        if not (res.success and res.status == 0):
            return False
        if np.any(res.x < -tolerance) or np.any(res.x > 1 + tolerance):
            return False
        if not np.isclose(np.sum(res.x), 1, atol=tolerance):
            return False
        return True
    
    def explore_blocked_distributions(self, blocked_distributions):
        value_restore_needed = False
        for node, distribution in blocked_distributions:
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


    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
        self.shield_calls += 1

        if trace_index >= len(self.current_nodes):
            self.current_nodes.append(self.initial_node)
            self.blocked_distributions.append([])

        if reset:
            self.trace_count += 1
            # we are looking at a different trace now, so we are going to update the shield we are using
            self.explore_blocked_distributions(self.blocked_distributions[trace_index])
            self.blocked_distributions[trace_index] = []
            self.current_nodes[trace_index] = self.initial_node
            last_action = None
        else:
            last_action = last_action[0]
            last_action_label = self.actions[last_action]

            prev_state_choice_labels = []
            for choice in range(self.model_info.model.transition_matrix.get_row_group_start(self.current_nodes[trace_index].state_index), self.model_info.model.transition_matrix.get_row_group_end(self.current_nodes[trace_index].state_index)):
                prev_state_choice_labels.append(self.model_info.model.choice_labeling.get_labels_of_choice(choice).pop())

            last_choice_index = prev_state_choice_labels.index(last_action_label)
            last_action = last_choice_index

            if (current_state, last_action) not in self.current_nodes[trace_index].successors.keys():
                self.current_nodes[trace_index].successors[(current_state, last_action)] = Node({}, self.current_nodes[trace_index], last_action, [], current_state, self.model_info.vmin[current_state])
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, last_action)]
            else:
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, last_action)]

        output_distribution = distribution

        # check if current distribution is inside the convex set
        if not self.point_in_convex_hull(self.current_nodes[trace_index].distributions + self.vmin_actions_distributions[current_state], distribution):
            self.blocked_actions += 1
            self.blocked_distributions[trace_index].append((self.current_nodes[trace_index], distribution))
            output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state])

        return output_distribution
    


# # TODO think about nice implementation where you could easily parameterize the behaviour anywhere between the two version of the self-constructing shield
class SelfConstructingShield(SelfConstructingShieldDeterministic):
    def __init__(self, model_info: ModelInfo, actions, nu: float, memory: int = 0):
        super().__init__(model_info, actions, nu, memory)

        self.last_distribution_indices = [None]

    # this can be optimized probably?
    def back_propagate_values(self, node: Node):
        while node is not None:
            # compute value from successors
            best_value = float('-inf')
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
            assert best_value != float('-inf')
            node.value = best_value
            node = node.predecessor

    def correct(self, last_action, current_state, distribution, reset, trace_index=0):
        self.shield_calls += 1

        if trace_index >= len(self.current_nodes):
            self.current_nodes.append(self.initial_node)
            self.last_distribution_indices.append(None)
            self.blocked_distributions.append([])

        if reset:
            self.trace_count += 1
            # we are looking at a different trace now, so we are going to update the shield we are using
            self.explore_blocked_distributions(self.blocked_distributions[trace_index])
            self.blocked_distributions[trace_index] = []
            self.current_nodes[trace_index] = self.initial_node
            self.last_distribution_indices[trace_index] = None
        else:
            # Change compared to parent class: last_played_action now stores the index of the played distribution
            if (current_state, self.last_distribution_indices[trace_index]) not in self.current_nodes[trace_index].successors.keys():
                all_dist = self.vmin_actions_distributions[self.current_nodes[trace_index].state_index] + self.current_nodes[trace_index].distributions
                assert self.last_distribution_indices[trace_index] is None and len(all_dist) == 0 or self.last_distribution_indices[trace_index] < len(all_dist)
                self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])] = Node({}, self.current_nodes[trace_index], self.last_distribution_indices[trace_index], [], current_state, self.model_info.vmin[current_state])
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]
            else:
                self.current_nodes[trace_index] = self.current_nodes[trace_index].successors[(current_state, self.last_distribution_indices[trace_index])]

        output_distribution = distribution

        # check if current distribution is inside the convex set
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