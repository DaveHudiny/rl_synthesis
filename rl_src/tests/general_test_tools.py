import numpy as np
from rl_src.environment.pomdp_builder import *

from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.environment.tf_py_environment import TFPyEnvironment


def init_environment(args : ArgsEmulator) -> tuple[EnvironmentWrapperVec, TFPyEnvironment]:
    prism_model = initialize_prism_model(args.prism_model, args.prism_properties, args.constants)
    env = EnvironmentWrapperVec(prism_model, args, num_envs=args.num_environments)
    tf_env = TFPyEnvironment(env)
    return env, tf_env

def init_args(prism_path, properties_path, nr_runs=101, goal_value_multiplier = 1.0, batched_vec_storm = False, masked_training = False, use_rnn_less=False,
              seed=None, max_steps=601, prefer_stochastic = True, stochastic_environment_actions=False) -> ArgsEmulator:
    args = ArgsEmulator(prism_model=prism_path, prism_properties=properties_path, learning_rate=1.6e-4,
                            restart_weights=0, learning_method="PPO", prefer_stochastic=prefer_stochastic,
                            nr_runs=nr_runs, agent_name="Testus", load_agent=False,
                            evaluate_random_policy=False, max_steps=max_steps, evaluation_goal=40.0, evaluation_antigoal=-0.0,
                            trajectory_num_steps=32, discount_factor=0.995, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=1000, random_start_simulator=False,
                            batch_size=256, vectorized_envs_flag=True, perform_interpretation=False, use_rnn_less=use_rnn_less, model_memory_size=0,
                            stacked_observations=False, batched_vec_storm=batched_vec_storm, masked_training=masked_training,
                            env_see_num_steps=False, env_see_last_action=False, env_see_reward=False, seed=seed)
    return args


def get_scalarized_reward(rewards, rewards_types):
    last_reward = rewards_types[-1]
    return rewards[last_reward]


def parse_properties(prism_properties: str) -> list[str]:
    with open(prism_properties, "r") as f:
        lines = f.readlines()
    properties = []
    for line in lines:
        if line.startswith("//"):
            continue
        properties.append(line.strip())
    return properties


def initialize_prism_model(prism_model: str, prism_properties, constants: dict[str, str]):
    properties = parse_properties(prism_properties)
    pomdp_args = POMDP_arguments(
        prism_model, properties, constants)
    return POMDP_builder.build_model(pomdp_args)


special_labels = np.array(["(((sched = 0) & (t = (8 - 1))) & (k = (20 - 1)))", "goal", "done", "((x = 2) & (y = 0))",
                           "((x = (10 - 1)) & (y = (10 - 1)))"])
