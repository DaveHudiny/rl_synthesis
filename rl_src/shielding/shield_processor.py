from rl_src.tools.args_emulator import ArgsEmulator
import tensorflow as tf

from rl_src.shielding.model_info import ModelInfo
import rl_src.shielding.shields
from rl_src.shielding.constructed_shield_data import ShieldData

import stormpy
import numpy as np

import pickle

class ShieldProcessor:
    def __init__(self, actions : list[str], model : stormpy.storage.SparsePomdp, nu : float, shield_type : str, args : ArgsEmulator = None, shield_memory : int = 0, debug: bool = False, shield_folder: str = None):
        self.args = args
        self.actions = actions
        self.shield_folder = shield_folder

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
            print("Max expected rewards to goal from initial state:", reward_result.get_values()[mdp.initial_states[0]])
            exit()

        observation_to_state = [None] * model.nr_observations
        for state in range(model.nr_states):
            obs = model.get_observation(state)
            observation_to_state[obs] = state

        assert None not in observation_to_state, "Some observations do not map to any state."
            
        model_info = ModelInfo(model=model, observation_to_state=observation_to_state, bad_state="bad", vmin=vmin, vmax=vmax)

        if shield_type == 'identity':
            self.shield = rl_src.shielding.shields.IdentityShield(model_info=model_info, actions=self.actions)
        elif shield_type == 'standard':
            self.shield = rl_src.shielding.shields.StandardShield(model_info=model_info, actions=self.actions)
        elif shield_type == 'pessimistic':
            self.shield = rl_src.shielding.shields.PessimisticShield(model_info=model_info, actions=self.actions, nu=nu)
        elif shield_type == 'optimistic':
            self.shield = rl_src.shielding.shields.OptimisticShield(model_info=model_info, actions=self.actions, nu=nu)
        elif shield_type == 'delta':
            self.shield = rl_src.shielding.shields.DeltaShield(model_info=model_info, actions=self.actions, delta=nu)
        elif shield_type == 'self-constructing-static':
            self.shield = rl_src.shielding.shields.SelfConstructingShield(model_info=model_info, actions=self.actions, nu=nu, memory=shield_memory)
        elif shield_type == 'self-constructing-safe':
            self.shield = rl_src.shielding.shields.SelfConstructingShieldConstructionSafe(model_info=model_info, actions=self.actions, nu=nu, memory=shield_memory)
        elif shield_type == 'self-constructing-unsafe':
            self.shield = rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe(model_info=model_info, actions=self.actions, nu=nu, memory=shield_memory)
        else:
            raise ValueError(f"Unknown shield type: {shield_type}")
        
        self.shield.rounding_precision = 6
        
        if self.shield_folder is not None:
            assert type(self.shield) in [rl_src.shielding.shields.SelfConstructingShieldConstructionSafe, rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe], "Saving shield can only be used with self-constructing shields."
        
    def save_shield(self, path: str, iteration = None):
        """Saves the shield to a file."""
        if not type(self.shield) in [rl_src.shielding.shields.SelfConstructingShieldConstructionSafe, rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe]:
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
        played_probs = tf.nn.softmax(played_logits).numpy().tolist()
        distributions = []

        for i in range(len(valuations)):

            current_state = self.shield.model_info.observation_to_state[integers[i][0]]
            mapped_played_distribution, current_state_choice_labels = self.map_played_distribution(played_probs[i], current_state)

            distribution = self.shield.correct(prev_actions[i], current_state, mapped_played_distribution, resets[i], i)

            distribution = [distribution[current_state_choice_labels.index(action)] if action in current_state_choice_labels else 0.0 for action in self.actions]

            distributions.append(distribution)

        distributions = np.array(distributions, dtype=np.float_)
        # Convert the boolean mask to logits.
        masked_logits = tf.math.log(distributions + 1e-10)
        return masked_logits

        
