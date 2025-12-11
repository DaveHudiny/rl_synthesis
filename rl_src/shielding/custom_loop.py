from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
import tensorflow as tf
import numpy as np

from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.environment.tf_py_environment import TFPyEnvironment

from rl_src.tests.general_test_tools import init_args, load_sketch
from rl_src.tools.evaluators import evaluate_policy_in_model
from rl_src.tools.evaluation_results_class import EvaluationResults
from rl_src.shielding.custom_policy import create_dummy_distribution, CustomPolicy

from rl_src.tools.trajectory_buffer import TrajectoryBuffer

from tf_agents.trajectories import Trajectory

def custom_loop(policy : TFPolicy, environment : EnvironmentWrapperVec, num_parallel_simulations: int, num_steps: int):
    environment.temporarily_set_num_envs(num_parallel_simulations)
    tf_environment = TFPyEnvironment(environment)

    trajectory_buffer = TrajectoryBuffer(environment)
    evaluation_result = EvaluationResults()
    
    # use tf_function for performance if needed -- remove, if the policy is not compatible with TF graph execution
    # policy_function = tf.function(policy.distribution)
    policy_function = policy.distribution
    # Time step is a structure that holds observation (triplet of observation, action mask, integer representing observation index), reward, step type and discount.
    time_step = tf_environment.reset()
    policy_state = policy.get_initial_state(batch_size=num_parallel_simulations)

    for step in range(num_steps):
        policy_step = policy_function(time_step, policy_state)
        distribution = policy_step.action
        policy_state = policy_step.state
        # Following operations represents identity, but you can modify it.
        probs = tf.nn.softmax(distribution.logits).numpy()
        # observation = time_step.observation["observation"].numpy().tolist()
        # mask = time_step.observation["mask"].numpy().tolist()
        # observation_integer = time_step.observation["observation_integer"].numpy().tolist()

        logits = tf.math.log(probs)
        # End of an identity block.



        action = tf.random.categorical(logits, 1, dtype=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        new_time_step = tf_environment.step(action)
        trajectory = Trajectory(
            step_type=time_step.step_type,
            observation=time_step.observation,
            action=action,
            policy_info=policy_step.info,
            reward=new_time_step.reward,
            discount=new_time_step.discount,
            next_step_type=new_time_step.step_type
        )
        trajectory_buffer.add_batched_step(trajectory)
        time_step = new_time_step

    trajectory_buffer.final_update_of_results(evaluation_result.update)
    evaluation_result.log_evaluation_info() # Ensure, that the number of finished episodes is at least 1!
    environment.reset_num_envs() # Sets the number of environments back to original value.

def test_custom_loop():
    project_path = "models/models_pomdp_no_family/network-3-8-20"

    args = init_args(prism_path=None, properties_path=None,
                     use_rnn_less=False, # Use RNN-less agent (if True, the policy should be completely memoryless)
                     max_steps=601, # Max steps per episode
                     seed=42, # Random seed, for the reproducibility, set it to some integer value
                     prefer_stochastic=True, # Whether to prefer stochastic or deterministic actions during the evaluation
                    )
    sketch = load_sketch(project_path=project_path)
    model = sketch.pomdp # If you don't have POMDP, you can switch to quotient mdp or some other MDP/POMDP representations.
    # model = sketch.quotient_mdp

    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True)

    tf_env = TFPyEnvironment(environment)
    custom_policy = CustomPolicy(
        action_spec=tf_env.action_spec(),
        time_step_spec=tf_env.time_step_spec(),
        distribution_function=create_dummy_distribution())

    custom_loop(
        policy=custom_policy,
        environment=environment,
        num_parallel_simulations=5,
        num_steps=500
    )
    