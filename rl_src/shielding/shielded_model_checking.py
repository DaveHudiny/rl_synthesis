import stormpy
import payntbind

import rl_src.shielding.shields
from rl_src.shielding.model_info import ModelInfo




def model_check_given_policy_and_shield(state_to_actions, shield, episode_length=50, goal_value=100.0, antigoal_value=-100.0, expected_shield_calls=False):
     
    if type(shield) in [rl_src.shielding.shields.OptimisticShield, rl_src.shielding.shields.PessimisticShield]:
        assert False, "Model checking not possible for optimistic or pessimistic shields."
    if type(shield) in [rl_src.shielding.shields.SelfConstructingShieldOnline, rl_src.shielding.shields.SelfConstructingShieldOffline]:
        assert False, "Model checking not possible for construction algorithms."

    model = shield.model_info.model

    episode_length_string = "" if episode_length is None else f"<={episode_length}"
    safety_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"bad\" ]")
    full_safety_formula = stormpy.parse_properties(f"P=? [ F \"bad\" ]")
    goal_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"goal\" ]")
    fail_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"fail\" ]")
    reward_formula = stormpy.parse_properties(f"R{{\"rews\"}}=? [ C{episode_length_string} ]")

    shield.shield_calls = 0
    shield.blocked_actions = 0

    if type(shield) in [rl_src.shielding.shields.IdentityShield, rl_src.shielding.shields.StandardShield, rl_src.shielding.shields.DeltaShield]:
        shielded_state_to_actions = []
        was_state_shielded = [0] * model.nr_states
        blocked_actions = 0
        for state, action in enumerate(state_to_actions):
            shielded_action = shield.correct(None, state, action, False)
            shielded_state_to_actions.append(shielded_action)
            if blocked_actions < shield.blocked_actions:
                was_state_shielded[state] = 1
                blocked_actions = shield.blocked_actions

        dtmc = payntbind.synthesis.applyRandomizedScheduler(shield.model_info.model, shielded_state_to_actions)

    if type(shield) in [rl_src.shielding.shields.SelfConstructingShield]:

        if shield.memory > 0:

            full_safety_formula = stormpy.parse_properties(f"Pmax=? [ F \"bad\" ]")

            was_state_shielded = [0] * (shield.memory_unfolded_model.nr_states)

            state_vector_matrix = []
            reward_model = shield.model_info.model.get_reward_model("rews")
            actual_rewards = []
            blocked_actions = 0
            for state in range(shield.memory_unfolded_model.nr_states):
                original_state = shield.memory_state_to_sliding_window[state][0]
                action = state_to_actions[original_state]
                shielded_action = shield.correct_for_given_memory_state(None, original_state, action, False, state)
                row = payntbind.synthesis.createCombinationOfRows(shield.full_transition_matrix_vector[state], shielded_action)
                state_vector_matrix.append(row)
                action_reward = 0.0
                for choice_index, prob in enumerate(shielded_action):
                    action_reward += prob * reward_model.state_action_rewards[shield.model_info.model.transition_matrix.get_row_group_start(original_state) + choice_index]
                actual_rewards.append(action_reward)

                if blocked_actions < shield.blocked_actions:
                    was_state_shielded[state] = 1
                    blocked_actions = shield.blocked_actions


            dtmc = payntbind.synthesis.createDtmcFromVectorMatrixWithRewards(shield.memory_unfolded_model, state_vector_matrix, actual_rewards)
        
        else:
        
            node_index_to_node = shield.initial_node.get_index_to_node_map()
            node_index_to_state = shield.initial_node.get_state_indices_map()
            node_index_to_successor_index_to_state = shield.initial_node.get_node_to_successor_index_to_state_map()

            shielded_state_to_actions = [None] * (len(node_index_to_node) + model.nr_states)
            successor_states_to_node = [None] * len(node_index_to_node)

            was_state_shielded = [0] * (len(node_index_to_node) + model.nr_states)

            standard_shield = rl_src.shielding.shields.StandardShield(shield.model_info, shield.actions)

            blocked_actions = 0
            for node_index, node in node_index_to_node.items():
                state = node.state_index
                action = state_to_actions[state]
                shielded_action, last_distr = shield.correct_for_given_node(None, state, action, False, node)

                shielded_state_to_actions[node_index] = shielded_action

                if blocked_actions < shield.blocked_actions:
                    assert tuple(shielded_action) != tuple(state_to_actions[state])
                    was_state_shielded[node_index] = 1
                    blocked_actions = shield.blocked_actions

                successor_state_to_node = {}
                for (succ_state, distr_index), succ_node in node_index_to_successor_index_to_state[node_index].items():
                    if distr_index == last_distr:
                        successor_state_to_node[succ_state] = succ_node
                successor_states_to_node[node_index] = successor_state_to_node

            for state in range(model.nr_states):
                shielded_action, _ = shield.correct_for_given_node(None, state, state_to_actions[state], False, None)
                shielded_state_to_actions[len(node_index_to_node) + state] = shielded_action

                if blocked_actions < shield.blocked_actions:
                    assert tuple(shielded_action) != tuple(state_to_actions[state])
                    was_state_shielded[len(node_index_to_node) + state] = 1
                    blocked_actions = shield.blocked_actions

            dtmc = payntbind.synthesis.applyRandomizedSchedulerFromTree(shield.model_info.model, shielded_state_to_actions, node_index_to_state, successor_states_to_node)



    safety_result = stormpy.model_checking(dtmc, safety_formula[0])
    full_safety_result = stormpy.model_checking(dtmc, full_safety_formula[0])
    if "goal" in model.labeling.get_labels():
        goal_result = stormpy.model_checking(dtmc, goal_formula[0])
    else:
        goal_result = None
    if "fail" in model.labeling.get_labels():
        fail_result = stormpy.model_checking(dtmc, fail_formula[0])
    else:
        fail_result = None
    reward_result = stormpy.model_checking(dtmc, reward_formula[0])

    # Compute expected shield calls and blocked actions
    if expected_shield_calls:
        unfolded_dtmc, state_to_orig_state_step_pairs = payntbind.synthesis.unfoldDtmcWithStepBound(dtmc, episode_length)
        import math
        environment = stormpy.Environment()
        result = stormpy.compute_expected_number_of_visits(environment, unfolded_dtmc)
        expected_visits_values = list(result.get_values())
        expected_visits_values = [v if v != float('inf') and not math.isnan(v) else 0.0 for v in expected_visits_values]
        # print(was_state_shielded)
        # print(expected_visits_values)

        # remove repeated visits of goal and fail states
        for unfolded_state, (orig_state, step) in enumerate(state_to_orig_state_step_pairs):
            if "goal" in dtmc.labeling.get_labels_of_state(orig_state) or "fail" in dtmc.labeling.get_labels_of_state(orig_state):
                expected_visits_values[unfolded_state] = 0.0

        num_expected_shield_calls = sum(expected_visits_values)
        num_expected_blocked_actions = 0.0
        earliest_shielded_step = None
        for unfolded_state, visits in enumerate(expected_visits_values):
            if unfolded_state >= len(state_to_orig_state_step_pairs):
                continue
            orig_state = state_to_orig_state_step_pairs[unfolded_state][0]
            if was_state_shielded[orig_state]:
                num_expected_blocked_actions += visits
                if earliest_shielded_step is None or state_to_orig_state_step_pairs[unfolded_state][1] < earliest_shielded_step:
                    earliest_shielded_step = state_to_orig_state_step_pairs[unfolded_state][1]
        # print(f"Expected blocked actions: {num_expected_blocked_actions}")
        # print(f"Expected shield calls: {num_expected_shield_calls}")
        # print(f"Allowed actions: {(1 - (num_expected_blocked_actions / num_expected_shield_calls)) * 100:.2f}%")
        # exit()
    else:
        num_expected_shield_calls = shield.shield_calls
        num_expected_blocked_actions = shield.blocked_actions
        earliest_shielded_step = None

    result_dict = {
        "safety_probability": safety_result.get_values()[dtmc.initial_states[0]],
        "full_safety_probability": full_safety_result.get_values()[dtmc.initial_states[0]],
        "goal_reachability": goal_result.get_values()[dtmc.initial_states[0]] if goal_result is not None else 'N/A',
        "fail_reachability": fail_result.get_values()[dtmc.initial_states[0]] if fail_result is not None else 'N/A',
        "expected_reward": reward_result.get_values()[dtmc.initial_states[0]],
        "actual_reward": (goal_value*goal_result.get_values()[dtmc.initial_states[0]] if goal_result is not None else 0) + (antigoal_value*fail_result.get_values()[dtmc.initial_states[0]] if fail_result is not None else 0) + reward_result.get_values()[dtmc.initial_states[0]],
        "expected_shield_calls": num_expected_shield_calls,
        "expected_blocked_actions": num_expected_blocked_actions,
        "earliest_shielded_step": earliest_shielded_step
    }

    print(f"Safety probability from initial state: {result_dict['safety_probability']}")
    print(f"Unbounded safety probability from initial state: {result_dict['full_safety_probability']}")
    print(f"Goal reachability from initial state: {result_dict['goal_reachability']}")
    print(f"Fail reachability from initial state: {result_dict['fail_reachability']}")
    print(f"Reward from initial state: {result_dict['expected_reward']}")
    print(f"Actual reward: {result_dict['actual_reward']}")
    print(f"Allowed actions: {(1 - (result_dict['expected_blocked_actions'] / result_dict['expected_shield_calls']))}")
    print(f"Earliest shielded step: {result_dict['earliest_shielded_step']}")
    
    return result_dict