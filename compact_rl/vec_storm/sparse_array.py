import chex
import numpy as np
from typing import Optional, Union
from itertools import product


@chex.dataclass
class SparseArray:
    nr_actions: int
    nr_states: int
    nr_rows: int
    row_starts: Union[chex.Array, np.ndarray]
    row_ends: Union[chex.Array, np.ndarray]
    indices: Union[chex.Array, np.ndarray]
    data: Union[chex.Array, np.ndarray]

    @classmethod
    def from_data(cls, nr_states, nr_actions, row_map, sparse_matrix: Optional["SparseArray"] = None):
        if sparse_matrix is None:
            return SparseArray(
                nr_actions = nr_actions,
                nr_states = nr_states,
                nr_rows = 0,
                row_starts = None,
                row_ends = None,
                indices = None,
                data = None,
            )

        nr_rows = 0
        row_starts = np.zeros(nr_states * nr_actions, dtype=np.int32)
        row_ends = np.zeros(nr_states * nr_actions, dtype=np.int32)
        for state, action in product(range(nr_states), range(nr_actions)):
            sa_idx = state * nr_actions + action
            row_idx = row_map[sa_idx]
            row_starts[sa_idx] = nr_rows
            if row_idx != -1:
                row = sparse_matrix.get_row(row_idx)
                nr_rows += len(row)
            row_ends[sa_idx] = nr_rows
        
        indices = np.zeros(nr_rows, dtype=np.int32)
        data = np.zeros(nr_rows, dtype=np.float32)
        for state, action in product(range(nr_states), range(nr_actions)):
            sa_idx = state * nr_actions + action
            row_idx = row_map[sa_idx]
            if row_idx == -1:
                continue
            row = sparse_matrix.get_row(row_idx)
            for i, entry in enumerate(row):
                indices[row_starts[sa_idx] + i] = entry.column
                data[row_starts[sa_idx] + i] = entry.value()
            
        return SparseArray(
            nr_actions = nr_actions,
            nr_states = nr_states,
            nr_rows = nr_rows,
            row_starts = row_starts,
            row_ends = row_ends,
            indices = indices,
            data = data,
        )

    @classmethod
    def zeros_like(cls, other, extra_shape=()):
        obj = cls.from_data(other.nr_states, other.nr_actions, None)
        obj.nr_rows = other.nr_rows
        obj.row_starts = other.row_starts.copy()
        obj.row_ends = other.row_ends.copy()
        obj.indices = other.indices.copy()
        obj.data = np.zeros(tuple(other.data.shape) + tuple(extra_shape), dtype=np.float32)
        return obj

    def get_row_range(self, state, action=None):
        idx = state if action is None else state * self.nr_actions + action
        return self.row_starts[idx], self.row_ends[idx]


    def get_element(sd, col_idx, state, action=None):
        idx = state if action is None else state * sd.nr_actions + action
        return sd.data[sd.row_starts[idx] + col_idx]

    def get_row_np(sd, state, action=None):
        idx = state if action is None else state * sd.nr_actions + action
        return sd.data[sd.row_starts[idx]:sd.row_ends[idx]]

    def get_row_indices_np(sd, state, action=None):
        idx = state if action is None else state * sd.nr_actions + action
        return sd.indices[sd.row_starts[idx]:sd.row_ends[idx]]
