from rl_src.tools.args_emulator import ArgsEmulator
import tensorflow as tf

import numpy as np

class ShieldProcessor:
    def __init__(self, nr_actions: int, args : ArgsEmulator = None):
        self.args = args
        self.nr_actions = nr_actions
        # TODO: Add the temporary shielding stuff.
        self.shielding_method = self.dummy_shield

    @staticmethod  
    def dummy_shield(state, prev_action, played_prob, resets, nr_actions):
        return played_prob

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
            distribution = self.shielding_method(integers[i], prev_actions[i], played_prob=played_probs[i], resets=resets[i], nr_actions=self.nr_actions)
            # If the distribution is larger than 0, True, otherwise False.
            distributions.append(distribution)
        
        distributions = np.array(distributions, dtype=np.float_)
        # Convert the boolean mask to logits.
        masked_logits = tf.math.log(distributions + 1e-10)
        return masked_logits

        
