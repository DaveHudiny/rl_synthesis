
import tensorflow as tf

from rl_src.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec

from paynt.quotient.fsc import FscFactored
from paynt.quotient.fsc import Fsc
from paynt.quotient.pomdp import PomdpQuotient
from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy

import numpy as np


class ConstructorFSC:
    """Class to construct a Finite State Controller (FSC) from different policies, e.g. TableBasedPolicy.
    This class provides a method to convert a table-based policy into an FSC, which can be used for reinforcement learning tasks.
    """

    @staticmethod
    def __cut_probs_1(action_for_memory, np_action_function, family_quotient_numpy: FamilyQuotientNumpy = None, original_action_labels: list[str] = None, memory=0, observation=0):
        most_probable_action = np.argmax(
            np_action_function[memory][observation])
        most_probable_action = int(most_probable_action) if (family_quotient_numpy and original_action_labels) is None else family_quotient_numpy.action_labels.tolist(
        ).index(original_action_labels[most_probable_action])
        if family_quotient_numpy is not None and not family_quotient_numpy.observation_to_legal_action_mask[observation][most_probable_action]:
            # If the most probable action is illegal, choose a random legal action
            legal_actions = [action for action in range(len(
                family_quotient_numpy.action_labels)) if family_quotient_numpy.observation_to_legal_action_mask[observation][action]]
            most_probable_action = np.random.choice(
                legal_actions)
        action_for_memory.append(int(most_probable_action))

    @staticmethod
    def _create_action_function(tf_action_function: tf.Tensor, family_quotient_numpy: FamilyQuotientNumpy = None, original_action_labels: list[str] = None, cut_probs=0.0):
        """Creates the action function for the FSC.
        Args:
            tf_action_function (tf.Tensor): The action function to be used in the FSC.
        Returns:
            tf.Tensor: The created action function.
        """
        np_action_function = tf.cast(
            tf_action_function, dtype=tf.float32).numpy()
        if np_action_function.shape[-1] == 1:
            # Deterministic action function
            return np_action_function.astype(int).tolist()
        else:
            # Convert the probabilites at the lowest level to dictionaries of selected action and its probability
            action_function = []
            for memory in range(np_action_function.shape[0]):
                action_for_memory = []
                for observation in range(np_action_function.shape[1]):
                    if cut_probs >= 1.0:
                        ConstructorFSC.__cut_probs_1(
                            action_for_memory, np_action_function, family_quotient_numpy, original_action_labels, memory, observation)
                        continue
                    action_dict = {}
                    illegal_action_prob = 0.0
                    for action, prob in enumerate(np_action_function[memory][observation]):
                        if prob > 0.0 and prob >= cut_probs:
                            action = int(action) if (family_quotient_numpy and original_action_labels) is None else family_quotient_numpy.action_labels.tolist(
                            ).index(original_action_labels[action])
                            if family_quotient_numpy is not None and not family_quotient_numpy.observation_to_legal_action_mask[observation][action]:
                                illegal_action_prob += prob
                                continue
                            else:
                                # To avoid zero probabilities
                                action_dict[action] = prob # + 0.0001
                        else:
                            action = int(action) if (family_quotient_numpy and original_action_labels) is None else family_quotient_numpy.action_labels.tolist(
                            ).index(original_action_labels[action])
                            if family_quotient_numpy is not None and family_quotient_numpy.observation_to_legal_action_mask[observation][action]:
                                # To avoid zero probabilities
                                # action_dict[action] = 0 # .0001
                                pass

                    if action_dict == {}:
                        if family_quotient_numpy is not None:
                            action_dict = {action: 1.0 for action in range(len(
                                family_quotient_numpy.action_labels)) if family_quotient_numpy.observation_to_legal_action_mask[observation][action]}
                            # Uniform distribution over legal actions
                            action_dict = {
                                action: 1.0 / len(action_dict) for action in action_dict}
                    # Uniformly distribute the illegal action probability to legal actions
                    if illegal_action_prob > 0.0:
                        for action in action_dict:
                            action_dict[action] += illegal_action_prob / len(
                                action_dict)

                    # Normalize the action probabilities
                    total_prob = sum(action_dict.values())
                    if total_prob > 0:
                        action_dict = {
                            action: prob / total_prob for action, prob in action_dict.items()}
                    else:
                        # If total_prob is 0, assign uniform distribution over legal actions
                        if family_quotient_numpy is not None:
                            action_dict = {action: 1.0 for action in range(len(
                                family_quotient_numpy.action_labels)) if family_quotient_numpy.observation_to_legal_action_mask[observation][action]}
                            action_dict = {
                                action: 1.0 / len(action_dict) for action in action_dict}
                    action_for_memory.append(action_dict)
                action_function.append(action_for_memory)
            return action_function

    @staticmethod
    def __create_update_function(tf_update_function: tf.Tensor, cut_probs=0.0):
        """Creates the update function for the FSC.
        Args:
            tf_update_function (tf.Tensor): The update function to be used in the FSC.
        Returns:
            tf.Tensor: The created update function.
        """

        np_update_function = tf.cast(
            tf_update_function, dtype=tf.float32).numpy()
        if np_update_function.shape[-1] == 1:  # Deterministic update
            # If there is only one update function, return it as a list
            # Deterministic update function
            return np_update_function.astype(int).tolist()
        else:
            # Convert the probabilites at the lowest level to dictionaries of selected update and its probability
            update_function = []
            for memory in range(np_update_function.shape[0]):
                update_for_memory = []
                for observation in range(np_update_function.shape[1]):
                    if cut_probs >= 1.0:
                        update_for_memory.append(
                            np.argmax(np_update_function[memory][observation]))
                        continue
                    update_dict = {}
                    for update, prob in enumerate(np_update_function[memory][observation]):
                        if prob > 0 and prob >= cut_probs:
                            update_dict[update] = prob
                    if update_dict == {}:
                        update_dict = {update: 1.0 for update in range(
                            np_update_function.shape[-1])}
                    # Normalize the update probabilities
                    total_prob = sum(update_dict.values())
                    if total_prob > 0:
                        update_dict = {
                            update: prob / total_prob for update, prob in update_dict.items()}

                    update_for_memory.append(update_dict)
                update_function.append(update_for_memory)
            return update_function

    @staticmethod
    def __create_observation_labels(pomdp_quotient: PomdpQuotient):
        return pomdp_quotient.observation_labels

    @staticmethod
    def __create_action_labels(table_based_policy: TableBasedPolicy):
        return list(table_based_policy.action_keywords)

    @staticmethod
    def __construct_factored_fsc(table_based_policy: TableBasedPolicy, pomdp_quotient: PomdpQuotient, family_quotient_numpy: FamilyQuotientNumpy = None, cut_probs=0.0) -> FscFactored:
        """Constructs a factored FSC from a table-based policy.
        Args:
            table_based_policy (TableBasedPolicy): The table-based policy to be converted into an FSC.
            pomdp_quotient (PomdpQuotient): The POMDP quotient used to create the FSC.
            family_quotient_numpy (FamilyQuotientNumpy): The family quotient used to create the FSC.
        Returns:
            FscFactored: The constructed FSC.
        """
        action_function = ConstructorFSC._create_action_function(
            table_based_policy.tf_observation_to_action_table, family_quotient_numpy, table_based_policy.action_keywords, cut_probs)
        update_function = ConstructorFSC.__create_update_function(
            table_based_policy.tf_observation_to_update_table, cut_probs)
        is_deterministic = True if type(
            action_function[0][0]) is int else False
        if cut_probs >= 1.0:
            is_deterministic = True
        try:
            observation_labels = ConstructorFSC.__create_observation_labels(
                pomdp_quotient)
        except:
            observation_labels = None
        action_labels = ConstructorFSC.__create_action_labels(
            table_based_policy) if family_quotient_numpy is None else family_quotient_numpy.action_labels.tolist()

        num_observations = len(action_function[0])
        num_nodes = len(action_function)

        # Create the FSC
        fsc = FscFactored(
            action_function=action_function,
            update_function=update_function,
            observation_labels=observation_labels,
            action_labels=action_labels,
            is_deterministic=is_deterministic,
            num_observations=num_observations,
            num_nodes=num_nodes,
        )
        return fsc

    def __construct_joint_fsc(table_based_policy: TableBasedPolicy, pomdp_quotient: PomdpQuotient, family_quotient_numpy: FamilyQuotientNumpy = None) -> Fsc:
        """Constructs a joint FSC from a table-based policy.
        Args:
            table_based_policy (TableBasedPolicy): The table-based policy to be converted into an FSC.
            pomdp_quotient (PomdpQuotient): The POMDP quotient used to create the FSC.
            family_quotient_numpy (FamilyQuotientNumpy): The family quotient used to create the FSC.
        Returns:
            Fsc: The constructed FSC.
        """
        np_transition_function = tf.cast(
            table_based_policy.joint_transition_function, dtype=tf.float32).numpy()
        if np_transition_function.shape[-1] == 1:  # Deterministic transition
            raise ValueError(
                "Joint transition function is deterministic, use a different method to construct the FSC.")
        else:
            fsc = Fsc(
                num_nodes=np_transition_function.shape[0],
                num_observations=np_transition_function.shape[1],
            )
            for memory in range(np_transition_function.shape[0]):
                for observation in range(np_transition_function.shape[1]):
                    action_dict = {}
                    for action, prob in enumerate(np_transition_function[memory][observation]):
                        memory_action = action // table_based_policy.mem_size
                        memory_update = action % table_based_policy.mem_size
                        memory_action = int(memory_action) if (family_quotient_numpy and table_based_policy.action_keywords) is None \
                            else family_quotient_numpy.action_labels.tolist().index(table_based_policy.action_keywords[memory_action])
                        if prob > 0.0:
                            if family_quotient_numpy is not None and not family_quotient_numpy.observation_to_legal_action_mask[observation][memory_action]:
                                continue
                            else:
                                action_dict[(
                                    memory_action, memory_update)] = prob
                    if action_dict == {}:
                        if family_quotient_numpy is not None:
                            action_dict = {(action, update): 1.0 for action in range(len(family_quotient_numpy.action_labels))
                                           if family_quotient_numpy.observation_to_legal_action_mask[observation][action] for update in range(table_based_policy.mem_size)}
                            action_dict = {
                                action: 1.0 / len(action_dict) for action in action_dict}
                    # Normalize the action probabilities
                    total_prob = sum(action_dict.values())
                    if total_prob > 0:
                        action_dict = {
                            action: prob / total_prob for action, prob in action_dict.items()}
                    fsc.transitions[memory][observation] = action_dict

        fsc.action_labels = ConstructorFSC.__create_action_labels(
            table_based_policy) if family_quotient_numpy is None else family_quotient_numpy.action_labels.tolist()

        return fsc

    @staticmethod
    def construct_fsc_from_table_based_policy(
        table_based_policy: TableBasedPolicy,
        pomdp_quotient: PomdpQuotient,
        family_quotient_numpy: FamilyQuotientNumpy,
        cut_probs=0.0

    ) -> FscFactored | Fsc:
        """Constructs a Finite State Controller (FSC) from a table-based policy.
        Args:
            table_based_policy (TableBasedPolicy): The table-based policy to be converted into an FSC.
            environment_wrapper (EnvironmentWrapperVec): The environment wrapper used to create the FSC.
        Returns:
            FSC: The constructed FSC.
        """
        # Create a new FSC object
        if table_based_policy.joint_transition_function is None:
            return ConstructorFSC.__construct_factored_fsc(table_based_policy, pomdp_quotient, family_quotient_numpy, cut_probs)
        else:
            return ConstructorFSC.__construct_joint_fsc(table_based_policy, pomdp_quotient, family_quotient_numpy)


def test_fsc_constructors():
    from robust_rl.robust_rl_tools import assignment_to_pomdp, load_sketch
    from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
    from tests.general_test_tools import init_args

    # Test the FSC construction from a table-based policy
    family_quotient = load_sketch("models/models_robust/network")

    family_quotient_numpy = FamilyQuotientNumpy(
        family_quotient=family_quotient)

    hole_assignment = family_quotient.family.pick_random()
    pomdp, _, _ = assignment_to_pomdp(family_quotient, hole_assignment)
    args = init_args("ahoj/sketch.templ", "ahoj/sketch.props")
    env = EnvironmentWrapperVec(pomdp, args, enforce_compilation=True,
                                obs_evaluator=family_quotient.obs_evaluator,
                                quotient_state_valuations=family_quotient.quotient_mdp.state_valuations,
                                observation_to_actions=family_quotient.observation_to_actions)

    # tf_action_function = np.random.uniform(
    #     low=0, high=1, size=[4, family_quotient_numpy.observation_to_legal_action_mask.shape[0], env.action_keywords])
    tf_action_function = tf.random.uniform(
        shape=[4, family_quotient_numpy.observation_to_legal_action_mask.shape[0], len(env.action_keywords)], minval=0, maxval=1, dtype=tf.float32)
    tf_action_function = tf_action_function.numpy()
    tf_action_function = tf_action_function.reshape(
        (4, family_quotient_numpy.observation_to_legal_action_mask.shape[0], len(env.action_keywords)))
    # Remove illegal actions from tf_action_function
    for obs in range(family_quotient_numpy.observation_to_legal_action_mask.shape[0]):
        for action in env.action_keywords:
            if not family_quotient_numpy.observation_to_legal_action_mask[obs][family_quotient_numpy.action_labels.tolist().index(action)]:
                tf_action_function[:, obs,
                                   env.action_keywords.index(action)] = 0.0
        if np.sum(tf_action_function[:, obs, :]) == 0.0:
            # If all actions are illegal, assign uniform distribution
            for action in range(len(env.action_keywords)):
                tf_action_function[:, obs, action] = 1.0 / \
                    len(env.action_keywords)
    tf_action_function /= np.sum(tf_action_function, axis=-1, keepdims=True)
    tf_action_function = tf.convert_to_tensor(
        tf_action_function, dtype=tf.float32)
