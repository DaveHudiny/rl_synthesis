"""Model info."""

from dataclasses import dataclass
import stormpy

@dataclass
class ModelInfo:
    """Information about a model."""

    model: stormpy.storage.SparsePomdp
    observation_to_state: list[int]
    bad_state: str
    vmin: list[float]
    vmax: list[float]
