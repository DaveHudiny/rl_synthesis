import stormpy # import tf_agents # The RL library
import tensorflow as tf

from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent # The recurrent PPO agent
from compact_rl.rl.tools.args_emulator import ArgsEmulator # The argument emulator (historically I supported command line arguments for all of them)
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec # The vectorized environment wrapper

from compact_rl.rl.tests.general_test_tools import *
from compact_rl.rl.tools.evaluators import evaluate_policy_in_model
from compact_rl.rl.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy 

import os

model_name = "geo-2-8"
prism_path = f"./rl_src/models/{model_name}" # Folder with the PRISM model, you can change this to your own path
prism_template = os.path.join(prism_path, "sketch.templ") # File with the prism template
prism_spec = os.path.join(prism_path, "sketch.props") # File with the task PTL specification

args = ArgsEmulator(
    prism_model=prism_template,
    prism_properties=prism_spec,
    constants="", # If PRISM model has some unresolved constants (such as Size=N, you can set them as "N=10")
    discount_factor=0.99,
    learning_rate=1.6e-4,
    trajectory_num_steps=32, # Number of steps in a single driver.run() call and the length of the trajectory (sub-episodes) for training the agent
    num_environments=256, # Number of environments to run in parallel/vectorized (batch size currently corresponds to this), I usually use 256
    batch_size=256,
    max_steps=400, # Number of steps, that can be taken in the environment per episode (if more, the episode is truncated)
    stacked_observations=True,
    use_rnn_less=True
)

stormpy_model = initialize_prism_model(args.prism_model, args.prism_properties, args.constants) # Initialize the PRISM model using stormpy
# stormpy_model = any POMDP storm model, if you have some constructed model -- it is useful in PAYNT, since PAYNT constructs its own POMDPs and I just take them.
print(stormpy_model) # Prints the description of the PRISM model

env = EnvironmentWrapperVec(stormpy_model, args, num_envs=args.num_environments)
print(env.vectorized_simulator.simulator.observation_labels)
tf_env = TFPyEnvironment(env)
print("Environment initialized")

# Example of how to use the environment
print(f"Initial TimeStep: {tf_env.reset()}")
print(f"TimeStep after some arbitrary step {tf_env.step(tf.zeros((args.num_environments,)))}") # The number of actions is equal to the number of environments
print(tf_env.time_step_spec())

agent = Recurrent_PPO_agent(
            env, tf_env, args)
agent.action(tf_env.reset()) # Example of how to use the policy

import  time
start = time.time()
agent.train_agent(iterations = 2000) # Train the agent for nr runs
end = time.time()
print(f"Training time {end-start}")
# I recommend to use at least 1000 iterations in general, but it depends on the task and the model

time_step = tf_env.reset() # Reset the environment before evaluation
agent.set_agent_greedy()
agent.set_policy_masking()

policy = agent.get_policy()
policy_state = policy.get_initial_state(tf_env.batch_size) # Get the initial state of the policy
step_infos = [time_step]
n_aut_learn_steps = 6000
for i in range(n_aut_learn_steps):
    policy_step = policy.action(time_step, policy_state) # Get the action from the policy
    # print(f"Policy step {i}: {policy_step.action}") # Print the action
    policy_state = policy_step.state # Update the policy state
    time_step = tf_env.step(policy_step.action) # Step the environment with the action
    step_infos.append((policy_step,time_step))

import pickle
with open(f"aut_learn_data_{model_name}.pkl", "wb") as fp:
    pickle.dump(step_infos,fp)


