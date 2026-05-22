from abc import ABC, abstractmethod
from compact_rl.rl.tools.args_emulator import ReplayBufferOptions


class AbstractAgent(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def train_agent(self, iterations: int, vectorized: bool = True, replay_buffer_option: ReplayBufferOptions = ReplayBufferOptions.ON_POLICY):
        raise NotImplementedError

    @abstractmethod
    def save_agent(self, path):
        raise NotImplementedError
