import stormpy
import payntbind

import rl_src.shielding.shields
from rl_src.shielding.model_info import ModelInfo




def model_check_given_policy_and_shield(state_to_actions, shield, episode_length=50):
     
    if type(shield) in [rl_src.shielding.shields.OptimisticShield, rl_src.shielding.shields.PessimisticShield]:
        assert False, "Model checking not possible for optimistic or pessimistic shields."
    if type(shield) in [rl_src.shielding.shields.SelfConstructingShieldConstructionSafe, rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe]:
        assert False, "Model checking not possible for construction algorithms."

    model = shield.model_info.model

    episode_length_string = "" if episode_length is None else f"<={episode_length}"
    safety_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"bad\" ]")
    full_safety_formula = stormpy.parse_properties(f"P=? [ F !\"bad\" ]")
    goal_formula = stormpy.parse_properties(f"P=? [ true U{episode_length_string} \"goal\" ]")
    reward_formula = stormpy.parse_properties(f"R{{\"rews\"}}=? [ C{episode_length_string} ]")

    if type(shield) in [rl_src.shielding.shields.IdentityShield, rl_src.shielding.shields.StandardShield, rl_src.shielding.shields.DeltaShield]:
        shielded_state_to_actions = []
        for state, action in enumerate(state_to_actions):
            shielded_action = shield.correct(None, state, action, False)
            shielded_state_to_actions.append(shielded_action)
        dtmc = payntbind.synthesis.applyRandomizedScheduler(shield.model_info.model, shielded_state_to_actions)

    if type(shield) in [rl_src.shielding.shields.SelfConstructingShield]:
        
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

        # print(dtmc.transition_matrix)

        # print(shield.initial_node)

        # print(shielded_state_to_actions)
        # print(successor_states_to_node)

    safety_result = stormpy.model_checking(dtmc, safety_formula[0])
    full_safety_result = stormpy.model_checking(dtmc, full_safety_formula[0])
    goal_result = stormpy.model_checking(dtmc, goal_formula[0])
    reward_result = stormpy.model_checking(dtmc, reward_formula[0])

    print(f"Safety probability from initial state: {safety_result.get_values()[dtmc.initial_states[0]]}")
    print(f"Unbounded safety probability from initial state: {full_safety_result.get_values()[dtmc.initial_states[0]]}")
    print(f"Goal reachability from initial state: {goal_result.get_values()[dtmc.initial_states[0]]}")
    print(f"Reward from initial state: {reward_result.get_values()[dtmc.initial_states[0]]}")
    print(f"Actual reward: {100*goal_result.get_values()[dtmc.initial_states[0]] + reward_result.get_values()[dtmc.initial_states[0]]}")
