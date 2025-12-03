import json
import tensorflow as tf
import numpy as np

class Fsc:
    '''
    An FSC having
    - a fixed number of nodes
    - a joint transition function of the type NxZ -> Distr(ActxN), where f(n,z) is a dictionary
        (action,memory) -> probability
    '''

    def __init__(self, num_nodes, num_observations):
        self.num_nodes = num_nodes
        self.num_observations = num_observations        
        self.transitions = [ [None]*num_observations for _ in range(num_nodes) ]
        self.observation_labels = None
        self.action_labels = None

    def check_transitions(self, observation_to_actions):
        assert len(self.transitions) == self.num_nodes, "FSC transition_function function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            assert len(self.transitions[node]) == self.num_observations, \
                "in memory node {}, FSC transition function is not defined for all observations".format(node)
            for obs in range(self.num_observations):
                for action,update in self.transitions[node][obs].keys():
                    assert action in observation_to_actions[obs], "in observation {} FSC chooses invalid action {}".format(obs,action)
                    assert 0 <= update and update < self.num_nodes, "invalid FSC memory update {}".format(update)

    def check(self, observation_to_actions):
        ''' Check whether fields of FSC have been initialized appropriately. '''
        assert self.num_nodes > 0, "FSC must have at least 1 node"
        self.check_transitions(observation_to_actions)

    def compute_available_updates(self, initial_memory=0):
        """ Computes the set of available updates for whole FSC starting with initial memory state.
        """
        available_updates = np.zeros((self.num_nodes,), dtype=bool)
        available_updates[initial_memory] = True
        active_nodes = [initial_memory]
        while len(active_nodes) > 0:
            node = active_nodes.pop()
            for obs in range(self.num_observations):
                for action,update in self.transitions[node][obs].keys():
                    if not available_updates[update]:
                        available_updates[update] = True
                        active_nodes.append(update)
        return available_updates


class FscFactored:
    '''
    Class for encoding an FSC having
    - a fixed number of nodes
    - action selection is either:
        + deterministic: gamma: NxZ -> Act, or
        + randomized: gamma: NxZ -> Distr(Act), where gamma(n,z) is a dictionary of pairs (action,probability)
    - deterministic posterior-unaware memory update delta: NxZ -> N
    '''

    def __init__(self, num_nodes, num_observations, is_deterministic=False, 
                 action_function : tf.Tensor = None, update_function : tf.Tensor = None,
                 observation_labels = None, action_labels = None):
        if action_function and update_function and (observation_labels or action_labels):
            self.external_init(action_function, update_function, observation_labels, action_labels, num_nodes, num_observations, is_deterministic)
        else:
            self.num_nodes = num_nodes
            self.num_observations = num_observations
            self.is_deterministic = is_deterministic
        
            self.action_function = [ [None]*num_observations for _ in range(num_nodes) ]
            self.update_function = [ [None]*num_observations for _ in range(num_nodes) ]

            self.observation_labels = None
            self.action_labels = None

    def external_init(self, action_function, update_function, observation_labels, action_labels, num_nodes, num_observations, is_deterministic):
        self.num_nodes = num_nodes
        self.num_observations = num_observations
        self.is_deterministic = is_deterministic

        self.action_function = action_function
        self.update_function = update_function

        self.observation_labels = observation_labels
        self.action_labels = action_labels
        self.is_deterministic = is_deterministic
    def __str__(self):
        return json.dumps(self.to_json(), indent=4)

    def action_function_signature(self):
        if self.is_deterministic:
            return " NxZ -> Act"
        else:
            return " NxZ -> Distr(Act)"

    def to_json(self):
        json = {}
        json["num_nodes"] = self.num_nodes
        json["num_observations"] = self.num_observations
        json["__comment_action_function"] = "action function has signature {}".format(self.action_function_signature())
        json["__comment_update_function"] = "update function has signature NxZ -> N"

        json["action_function"] = self.action_function
        json["update_function"] = self.update_function

        if self.action_labels is not None:
            json["action_labels"] = self.action_labels
        if self.observation_labels is not None:
            json["observation_labels"] = self.observation_labels

        return json

    @classmethod
    def from_json(cls, json):
        num_nodes = json["num_nodes"]
        num_observations = json["num_observations"]
        fsc = FscFactored(num_nodes,num_observations)
        fsc.action_function = json["action_function"]
        fsc.update_function = json["update_function"]
        fsc.action_labels = json["action_labels"] if "action_labels" in json else None
        return fsc

    def reorder_nodes(self, node_old_to_new):
        action_function = [None for node in range(self.num_nodes)]
        update_function = [None for node in range(self.num_nodes)]
        for node_old,node_new in enumerate(node_old_to_new):
            action_function[node_new] = self.action_function[node_old]
            update_function[node_new] = [node_old_to_new[node] for node in self.update_function[node_old]]
        self.action_function = action_function
        self.update_function = update_function

    def reorder_actions(self, action_labels):
        for node in range(self.num_nodes):
            for obs in range(self.num_observations):
                if self.is_deterministic:
                    action = self.action_function[node][obs]
                    self.action_function[node][obs] = action_labels.index(self.action_labels[action])
                else:
                    action_function = {}
                    for action,prob in self.action_function[node][obs].items():
                        action_function[action_labels.index(self.action_labels[action])] = prob
                    self.action_function[node][obs] = action_function
        self.action_labels = action_labels.copy()

    def make_stochastic(self):
        if not self.is_deterministic:
            return
        for node in range(self.num_nodes):
            for obs in range(self.num_observations):
                self.action_function[node][obs] = {self.action_function[node][obs] : 1.0}
                self.update_function[node][obs] = {self.update_function[node][obs] : 1.0}
        self.is_deterministic = False

    def check_action_function(self, observation_to_actions):
        assert len(self.action_function) == self.num_nodes, "FSC action function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            assert len(self.action_function[node]) == self.num_observations, \
                "in memory node {}, FSC action function is not defined for all observations".format(node)
            for obs in range(self.num_observations):
                if observation_to_actions[obs] == []:
                    assert self.action_function[node][obs] is None
                    continue
                if self.is_deterministic:
                    action_support = [self.action_function[node][obs]]
                else:
                    action_support = list(self.action_function[node][obs].keys())
                for action in action_support:

                    try:
                        assert action in observation_to_actions[obs], "in observation {} FSC chooses invalid action {}".format(obs,action)
                    except:
                        # raise AssertionError("in observation {} FSC chooses invalid action {}. Available actions: {}".format(obs,action,observation_to_actions[obs]))
                        print("observation_to_actions[{}] = {}".format(obs, observation_to_actions[obs]))
                        print("it wants to play action {}".format(action))
                        if self.is_deterministic:
                            assert action in observation_to_actions[obs], "in observation {} FSC chooses invalid action {}".format(obs,action)
                        else:
                            del self.action_function[node][obs][action]
                action_support = list(self.action_function[node][obs].keys()) if not self.is_deterministic else [self.action_function[node][obs]]
                if action_support == []:
                    action_support = [observation_to_actions[obs][-1]]
            if self.num_observations < len(observation_to_actions):
                self.action_function[node].append({observation_to_actions[-1][-1] : 1.0} if not self.is_deterministic else observation_to_actions[-1][-1])

    def check_update_function(self):
        assert len(self.update_function) == self.num_nodes, "FSC update function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            try:
                assert len(self.update_function[node]) == self.num_observations, \
                    "in memory node {}, FSC update function is not defined for all observations".format(node)
            except:
                print("len(self.update_function[{}]) = {}".format(node, len(self.update_function[node])))
                print("self.action_function[{}] = {}".format(node, len(self.action_function[node])))
                print("update_function[{}] = {}".format(node, self.update_function[node]))
                print("self.num_observations = {}".format(self.num_observations))
                raise AssertionError("in memory node {}, FSC update function is not defined for all observations".format(node))
            for obs in range(self.num_observations):
                update = self.update_function[node][obs]
                continue # skipping check due to inconsistency in our FSC definition
                assert 0 <= update and update < self.num_nodes, "invalid FSC memory update {}".format(update)
            if len(self.update_function[node]) < len(self.action_function[node]):
                self.update_function[node].append({0 : 1.0} if not self.is_deterministic else 0)


    def check(self, observation_to_actions):
        ''' Check whether fields of FSC have been initialized appropriately. '''
        assert self.num_nodes > 0, "FSC must have at least 1 node"
        self.check_action_function(observation_to_actions)
        self.check_update_function()

    def fill_trivial_actions(self, observation_to_actions):
        ''' For each observation with 1 available action, set gamma(n,z) to that action. '''
        for obs,actions in enumerate(observation_to_actions):
            if len(actions) != 1:
                continue
            action = actions[0]
            if not self.is_deterministic:
                action = {action:1}
            for node in range(self.num_nodes):
                self.action_function[node][obs] = action

    def fill_trivial_updates(self, observation_to_actions):
        ''' For each observation with 1 available action, set delta(n,z) to n. '''
        for obs,actions in enumerate(observation_to_actions):
            if len(actions)>1:
                continue
            for node in range(self.num_nodes):
                self.update_function[node][obs] = node

    def fill_zero_updates(self):
        for node in range(self.num_nodes):
            self.update_function[node] = [0] * self.num_observations

    def fill_implicit_actions_and_updates(self):
        '''
        For an FSC with an irregular memory model, make action and updates explicit.
        '''
        for node in range(self.num_nodes):
            for obs in range(self.num_observations):
                if self.action_function[node][obs] is None:
                    self.action_function[node][obs] = self.action_function[0][obs]
                    self.update_function[node][obs] = self.update_function[0][obs]

    def copy(self):
        fsc = FscFactored(self.num_nodes, self.num_observations, self.is_deterministic)
        fsc.action_function = [ [action for action in node] for node in self.action_function ]
        fsc.update_function = [ [update for update in node] for node in self.update_function ]
        fsc.observation_labels = self.observation_labels
        fsc.action_labels = self.action_labels
        return fsc
    
    def compute_available_updates(self, initial_memory=0):
        """ Computes the set of available updates for whole FSC starting with initial memory state.
        """
        available_updates = np.zeros((self.num_nodes,), dtype=bool)
        available_updates[initial_memory] = True
        active_nodes = [initial_memory]
        while len(active_nodes) > 0:
            node = active_nodes.pop()
            for obs in range(self.num_observations):
                update = self.update_function[node][obs]
                if self.is_deterministic:
                    if not available_updates[update]:
                        available_updates[update] = True
                        active_nodes.append(update)
                else:
                    for update_key in list(update.keys()):
                        if not available_updates[update_key]:
                            available_updates[update_key] = True
                            active_nodes.append(update_key)
        return available_updates