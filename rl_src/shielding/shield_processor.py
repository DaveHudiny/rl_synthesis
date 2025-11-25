from tools.args_emulator import ArgsEmulator
import tensorflow as tf

from shielding.model_info import ModelInfo
import shielding.shields

import stormpy
import numpy as np

class ShieldProcessor:
    def __init__(self, actions : int, model : stormpy.storage.SparsePomdp, nu : float, shield_type : str, args : ArgsEmulator = None):
        self.args = args
        self.actions = actions

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

        # model checking results for debugging
        # reach_formula = stormpy.parse_properties("Pmax=? [ F \"goal\" ]")
        # reward_formula = stormpy.parse_properties("Rmax=? [ F \"goal\" ]")
        # reach_result = stormpy.model_checking(mdp, reach_formula[0])
        # reward_result = stormpy.model_checking(mdp, reward_formula[0])
        # print("Max reachability probabilities to goal from initial state:", reach_result.get_values()[mdp.initial_states[0]])
        # print("Max expected rewards to goal from initial state:", reward_result.get_values()[mdp.initial_states[0]])
        # exit()


        observation_to_state = [None] * model.nr_observations
        for state in range(model.nr_states):
            obs = model.get_observation(state)
            observation_to_state[obs] = state

        assert None not in observation_to_state, "Some observations do not map to any state."
            
        model_info = ModelInfo(model=model, observation_to_state=observation_to_state, bad_state="bad", vmin=vmin, vmax=vmax)

        if shield_type == 'identity':
            self.shield = shielding.shields.IdentityShield(model_info=model_info, actions=self.actions)
        elif shield_type == 'standard':
            self.shield = shielding.shields.StandardShield(model_info=model_info, actions=self.actions)
        elif shield_type == 'pessimistic':
            self.shield = shielding.shields.PessimisticShield(model_info=model_info, actions=self.actions, nu=nu)
        elif shield_type == 'optimistic':
            self.shield = shielding.shields.OptimisticShield(model_info=model_info, actions=self.actions, nu=nu)
        elif shield_type == 'self-constructing':
            self.shield = shielding.shields.SelfConstructingShieldDistributions(model_info=model_info, actions=self.actions, nu=nu)
        elif shield_type == 'self-constructing-simple':
            self.shield = shielding.shields.SelfConstructingShield(model_info=model_info, actions=self.actions, nu=nu)
        else:
            raise ValueError(f"Unknown shield type: {shield_type}")


    @staticmethod  
    def dummy_shield(state, prev_action, played_prob, resets, nr_actions):
        return played_prob
    
    def fix_distribution(self, distribution):
        total_prob = sum(distribution)
        if total_prob > 0:
            return [prob / total_prob for prob in distribution]
        else:
            uniform_prob = 1.0 / len(distribution)
            return [uniform_prob for _ in distribution]

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
        # print(len(valuations))
        # exit()

        for i in range(len(valuations)):

            current_state = self.shield.model_info.observation_to_state[integers[i][0]]
            current_state_choice_labels = []

            for choice in range(self.shield.model_info.model.transition_matrix.get_row_group_start(current_state), self.shield.model_info.model.transition_matrix.get_row_group_end(current_state)):
                current_state_choice_labels.append(self.shield.model_info.model.choice_labeling.get_labels_of_choice(choice).pop())

            mapped_played_distribution = [played_probs[i][self.actions.index(action)] for action in current_state_choice_labels]
            mapped_played_distribution = self.fix_distribution(mapped_played_distribution)

            distribution = self.shield.correct(prev_actions[i], current_state, mapped_played_distribution, resets[i])

            distribution = [distribution[current_state_choice_labels.index(action)] if action in current_state_choice_labels else 0.0 for action in self.actions]

            # If the distribution is larger than 0, True, otherwise False.
            distributions.append(distribution)
        
        distributions = np.array(distributions, dtype=np.float_)
        # Convert the boolean mask to logits.
        masked_logits = tf.math.log(distributions + 1e-10)
        return masked_logits

        
