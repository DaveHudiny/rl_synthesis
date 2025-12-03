from typing import Union, List, Tuple

import tf_agents
import tensorflow_probability as tfp

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec

from rl_src.tools.trajectory_buffer import TrajectoryBuffer

def sample_data_with_policy(policy: TFPolicy, num_samples=100,
                            environment: EnvironmentWrapperVec = None,
                            tf_environment: TFPyEnvironment = None,
                            trajectory_buffer: TrajectoryBuffer = None,
                            use_replay_buffer=True,
                            get_both=False,
                            seed=None) -> Union[TFUniformReplayBuffer,List, Tuple[List, TFUniformReplayBuffer]]:
    prev_time_step = tf_environment.reset()
    seed_stream = tfp.util.SeedStream(seed, salt="sampling")
    policy_state = policy.get_initial_state(environment.batch_size)

    action_function = tf_agents.utils.common.function(policy.action)
    if get_both:
        replay_buffer_lists = [prev_time_step]
        replay_buffer_buffer = TFUniformReplayBuffer(
            data_spec=policy.trajectory_spec, batch_size=environment.batch_size, max_length=num_samples+1)
        replay_buffer = [replay_buffer_lists, replay_buffer_buffer]
    elif use_replay_buffer:
        replay_buffer = TFUniformReplayBuffer(
            data_spec=policy.trajectory_spec, batch_size=environment.batch_size, max_length=num_samples+1)
    else:
        replay_buffer = [prev_time_step]

    for i in range(num_samples):
        policy_step = action_function(prev_time_step, policy_state, seed=seed_stream())
        action = policy_step.action
        policy_state = policy_step.state
        time_step = tf_environment.step(action)
        if get_both:
            replay_buffer_lists.append((policy_step, time_step))
            traj = tf_agents.trajectories.trajectory.from_transition(
                prev_time_step, policy_step, time_step)
            replay_buffer_buffer.add_batch(traj)
            if trajectory_buffer is not None:
                trajectory_buffer.add_batched_step(traj)
        elif use_replay_buffer:
            traj = tf_agents.trajectories.trajectory.from_transition(
                prev_time_step, policy_step, time_step)
            replay_buffer.add_batch(traj)
            if trajectory_buffer is not None:
                trajectory_buffer.add_batched_step(
                    traj)
        else:
            replay_buffer.append((policy_step, time_step))
        prev_time_step = time_step
    return replay_buffer

