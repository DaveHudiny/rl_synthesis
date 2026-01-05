import stormpy
import payntbind

import rl_src.shielding.shields
from rl_src.shielding.model_info import ModelInfo




def model_check_given_policy_and_shield(state_to_actions, shield, episode_length=50, goal_value=100.0, antigoal_value=-100.0):
     
    if type(shield) in [rl_src.shielding.shields.OptimisticShield, rl_src.shielding.shields.PessimisticShield]:
        assert False, "Model checking not possible for optimistic or pessimistic shields."
    if type(shield) in [rl_src.shielding.shields.SelfConstructingShieldConstructionSafe, rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe]:
        assert False, "Model checking not possible for construction algorithms."

    model = shield.model_info.model

    episode_length_string = "" if episode_length is None else f"<={episode_length}"
    safety_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"bad\" ]")
    full_safety_formula = stormpy.parse_properties(f"P=? [ F \"bad\" ]")
    goal_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"goal\" ]")
    fail_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"fail\" ]")
    reward_formula = stormpy.parse_properties(f"R{{\"rews\"}}=? [ C{episode_length_string} ]")

    if type(shield) in [rl_src.shielding.shields.IdentityShield, rl_src.shielding.shields.StandardShield, rl_src.shielding.shields.DeltaShield]:
        shielded_state_to_actions = []
        for state, action in enumerate(state_to_actions):
            shielded_action = shield.correct(None, state, action, False)
            shielded_state_to_actions.append(shielded_action)
        dtmc = payntbind.synthesis.applyRandomizedScheduler(shield.model_info.model, shielded_state_to_actions)

    if type(shield) in [rl_src.shielding.shields.SelfConstructingShield]:

        if shield.memory > 0:

            full_safety_formula = stormpy.parse_properties(f"Pmax=? [ F \"bad\" ]")

            state_vector_matrix = []
            reward_model = shield.model_info.model.get_reward_model("rews")
            actual_rewards = []
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


            dtmc = payntbind.synthesis.createDtmcFromVectorMatrixWithRewards(shield.memory_unfolded_model, state_vector_matrix, actual_rewards)
        
        else:
        
            node_index_to_node = shield.initial_node.get_index_to_node_map()
            node_index_to_state = shield.initial_node.get_state_indices_map()
            node_index_to_successor_index_to_state = shield.initial_node.get_node_to_successor_index_to_state_map()

            shielded_state_to_actions = [None] * (len(node_index_to_node) + model.nr_states)
            successor_states_to_node = [None] * len(node_index_to_node)

            for node_index, node in node_index_to_node.items():
                state = node.state_index
                action = state_to_actions[state]
                shielded_action, last_distr = shield.correct_for_given_node(None, state, action, False, node)

                shielded_state_to_actions[node_index] = shielded_action

                successor_state_to_node = {}
                for (succ_state, distr_index), succ_node in node_index_to_successor_index_to_state[node_index].items():
                    if distr_index == last_distr:
                        successor_state_to_node[succ_state] = succ_node
                successor_states_to_node[node_index] = successor_state_to_node

            for state in range(model.nr_states):
                shielded_action, _ = shield.correct_for_given_node(None, state, state_to_actions[state], False, None)
                shielded_state_to_actions[len(node_index_to_node) + state] = shielded_action

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

        result_dict = {
            "safety_probability": safety_result.get_values()[dtmc.initial_states[0]],
            "full_safety_probability": full_safety_result.get_values()[dtmc.initial_states[0]],
            "goal_reachability": goal_result.get_values()[dtmc.initial_states[0]] if goal_result is not None else 'N/A',
            "fail_reachability": fail_result.get_values()[dtmc.initial_states[0]] if fail_result is not None else 'N/A',
            "expected_reward": reward_result.get_values()[dtmc.initial_states[0]],
            "actual_reward": (goal_value*goal_result.get_values()[dtmc.initial_states[0]] if goal_result is not None else 0) + (antigoal_value*fail_result.get_values()[dtmc.initial_states[0]] if fail_result is not None else 0) + reward_result.get_values()[dtmc.initial_states[0]] 
        }

        print(f"Safety probability from initial state: {result_dict['safety_probability']}")
        print(f"Unbounded safety probability from initial state: {result_dict['full_safety_probability']}")
        print(f"Goal reachability from initial state: {result_dict['goal_reachability']}")
        print(f"Reward from initial state: {result_dict['expected_reward']}")
        print(f"Actual reward: {result_dict['actual_reward']}")
    
    return result_dict