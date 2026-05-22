import numpy as np


class Reorganizer:

    @staticmethod
    def compute_new_memory_mapping(initial_memory: int, update_function: np.ndarray) -> dict[int, int]:
        already_activated_memory_nodes = np.zeros(
            update_function.shape[0], dtype=bool)
        active_memory_nodes = [initial_memory]
        nr_used_memory_nodes = 0
        memory_to_new_index_dict = {}
        # Compute all active memory nodes and their new indices
        while len(active_memory_nodes) > 0:
            current_memory_node = active_memory_nodes.pop(0)
            memory_to_new_index_dict[current_memory_node] = nr_used_memory_nodes
            nr_used_memory_nodes += 1
            already_activated_memory_nodes[current_memory_node] = True
            # Create array of next available memory nodes from the current memory node and any observation
            available_updates = update_function[current_memory_node,
                                                :].reshape(-1)
            available_updates = np.unique(available_updates).tolist()
            for next_memory_node in available_updates:
                if not already_activated_memory_nodes[next_memory_node]:
                    active_memory_nodes.append(next_memory_node)
                    already_activated_memory_nodes[next_memory_node] = True
        return memory_to_new_index_dict

    @staticmethod
    def reorganize_action_and_update_functions(action_function: np.ndarray, update_function: np.ndarray, initial_memory: int):
        """ FSC extracted from quantized bottleneck extractor has different initial memory than the 0. 
        This function reorganizes the action and update functions to match the initial memory and available memory nodes.

        This function also removes the unused memory nodes and thus makes the controller smaller. Works only for deterministic (dirac) updates
        """
        is_stochastic_update = False
        if len(update_function.shape) == 3:
            # If the update function is 3D, we assume it is a stochastic form of update function with dirac probabilities.
            is_stochastic_update = True
            update_function = np.argmax(update_function, axis=2)

        memory_to_new_index_dict = Reorganizer.compute_new_memory_mapping(
            initial_memory, update_function)
        new_memory_size = len(memory_to_new_index_dict)
        new_action_function = np.zeros(
            (new_memory_size, action_function.shape[1], action_function.shape[2]), dtype=np.float32)
        new_update_function = np.zeros(
            (new_memory_size, update_function.shape[1],), dtype=np.int32)
        if is_stochastic_update:
            # If the update function is stochastic, we need to convert it back to the 3D format
            new_update_function_stochastic = np.zeros(
                (new_memory_size, update_function.shape[1], new_memory_size), dtype=np.float32)
        for old_memory_node, new_memory_node in memory_to_new_index_dict.items():
            new_action_function[new_memory_node, :,
                                :] = action_function[old_memory_node, :, :]
            new_update_function[new_memory_node, :] = [
                memory_to_new_index_dict[update] for update in update_function[old_memory_node, :]]
            if is_stochastic_update:
                # If the update function is stochastic, we need to convert it back to the 3D format
                new_update_function_stochastic[new_memory_node, :] = np.eye(
                    new_memory_size)[new_update_function[new_memory_node, :]]

        if is_stochastic_update:
            return new_action_function, new_update_function_stochastic
        else:
            return new_action_function, new_update_function


def test():
    action_function = np.array([[[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.0, 0.2, 0.8]],
                                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                                [[0.45, 0.55, 0.0], [0.3, 0.3, 0.4], [0.0, 0.5, 0.5]],
                                [[0.88, 0.12, 0.0], [0.7, 0.3, 0.0], [0.0, 0.6, 0.4]]])

    # No update goes to memory 1
    update_function = np.array([[0, 2, 3], [0, 2, 3], [0, 0, 3], [0, 3, 2]])
    update_function_dirac = np.array([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                      [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                      [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
                                      [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]])
    initial_memory = 2
    new_action_function, new_update_function = Reorganizer.reorganize_action_and_update_functions(
        action_function, update_function, initial_memory)
    print("New Action Function:")
    print(new_action_function)
    print("New Update Function:")
    print(new_update_function)
    new_action_function, new_update_function = Reorganizer.reorganize_action_and_update_functions(
        action_function, update_function_dirac, initial_memory)
    print("New Action Function (Dirac):")
    print(new_action_function)
    print("New Update Function (Dirac):")
    print(new_update_function)


if __name__ == "__main__":
    test()
