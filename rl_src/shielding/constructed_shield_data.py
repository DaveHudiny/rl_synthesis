"""Constructed shield data."""

from dataclasses import dataclass
from rl_src.shielding.shields import Node


@dataclass
class ShieldData:
    """Information about the constructed shield."""

    actions: list[str]
    original_model_nr_states: int
    observation_to_state: list[int]
    memory: int
    rounding_precision: int
    
    initial_node: Node # no memory bound shield
    current_action_distributions : list[list[float]] # memory bounded shield