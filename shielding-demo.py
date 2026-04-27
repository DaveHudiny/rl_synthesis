from rl_src.shielding.model_info import ModelInfo
import rl_src.shielding.shields

import stormpy
import stormpy.simulator

import click

import numpy as np


def get_builder_options():
    builder_options = stormpy.BuilderOptions()
    builder_options.set_build_state_valuations(True)
    builder_options.set_build_with_choice_origins(True)
    builder_options.set_build_all_labels(True)
    builder_options.set_build_choice_labels(True)
    builder_options.set_add_overlapping_guards_label(True)
    builder_options.set_build_observation_valuations(True)
    builder_options.set_build_all_reward_models(True)

    return builder_options


def get_model_info(model):

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
    # print("Vmin and Vmax for initial state:", vmin[mdp.initial_states[0]], vmax[mdp.initial_states[0]])

    observation_to_state = [None] * model.nr_observations
    for state in range(model.nr_states):
        obs = model.get_observation(state)
        observation_to_state[obs] = state

    model_info = ModelInfo(model=model, observation_to_state=observation_to_state, bad_state="bad", vmin=vmin, vmax=vmax)

    return model_info



@click.command()
@click.argument('model_path', type=click.Path(exists=True), default='models/shielding/test-corridor/sketch.templ')
@click.option("--shield-type", type=click.Choice(['safe', 'pessimistic', 'optimistic', 'delta', 'offline', 'online']), default='safe', help="Shielding method to use.")
@click.option("--nu", type=float, required=False, default=0.05, help="Nu (safety) value to use for shielding.")
def shielding_demo(model_path, shield_type, nu):

    # MODEL INITIALIZATION

    prism = stormpy.parse_prism_program(model_path)
    builder_options = get_builder_options()
    model = stormpy.build_sparse_model_with_options(prism, builder_options)

    assert model.nr_states == model.nr_observations, "We currently only support shielding for MDPs."
    assert model.initial_states is not None and len(model.initial_states) == 1, "We currently only support single initial state models."

    model_info = get_model_info(model)
    actions = model.choice_labeling.get_labels()


    # SHIELD INITIALIZATION

    if shield_type == 'safe':
        shield = rl_src.shielding.shields.StandardShield(model_info=model_info, actions=actions)
    elif shield_type == 'pessimistic':
        shield = rl_src.shielding.shields.PessimisticShield(model_info=model_info, actions=actions, nu=nu)
    elif shield_type == 'optimistic':
        shield = rl_src.shielding.shields.OptimisticShield(model_info=model_info, actions=actions, nu=nu)
    elif shield_type == 'delta':
        shield = rl_src.shielding.shields.DeltaShield(model_info=model_info, actions=actions, delta=nu)
    elif shield_type == 'online':
        shield = rl_src.shielding.shields.SelfConstructingShieldOnline(model_info=model_info, actions=actions, nu=nu)
    elif shield_type == 'offline':
        shield = rl_src.shielding.shields.SelfConstructingShieldOffline(model_info=model_info, actions=actions, nu=nu)
    else:
        raise ValueError(f"Unknown shield type: {shield_type}")
    
    shield.rounding_precision = 6


    # RUNNING SHIELD

    # you can call shield.correct() to get the shielded distribution
    # last_action_index - int
    # current_state_index - int
    # action_distribution - list of floats, the length of this list should match the number of available actions in the current state, and the values should sum to 1.0
    # reset - bool, whether this is the first step in a new episode (i.e. the current state is the initial state)
    # shielded_distribution = shield.correct(last_action_index, current_state_index, action_distribution, reset)

    simulator = stormpy.simulator.create_simulator(model)

    bad_episodes = 0
    for episode in range(100):
        # simulate an episode
        current_state, _prob, _labels = simulator.restart()
        last_action = None
        for step in range(100):
            # get available actions in the current state
            nr_available_actions = model.get_nr_available_actions(current_state)
            # uniform distribution over available actions
            action_distribution = [1.0 / nr_available_actions for _ in range(nr_available_actions)]

            print(f"Step {step}, State {current_state}, Action distribution before shielding: {action_distribution}")

            # get shielded distribution
            shielded_distribution = shield.correct(last_action, current_state, action_distribution, step == 0)

            print(f"Current node: {shield.current_nodes[0].node_index}, {shield.current_nodes[0].state_index}, {shield.current_nodes[0].value}, {shield.current_nodes[0].distributions}")
            print(f"Action distribution after shielding: {shielded_distribution}")

            # sample an action from the shielded distribution
            action_index = np.random.choice(len(actions), p=shielded_distribution)
            last_action = action_index

            next_state, _prob, labels = simulator.step(action_index)

            # check if we reached a bad state (for demonstration purposes, we assume that any state labeled "bad" is a bad state)
            if "bad" in labels:
                print(f"Reached a bad state in episode {episode}")
                bad_episodes += 1
                break

            current_state = next_state

    print(f"Number of episodes that reached a bad state: {bad_episodes} out of 100")




if __name__ == "__main__":
    shielding_demo()