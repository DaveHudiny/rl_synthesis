import pickle
from rl_src.shielding.shielding_options import ShieldingOptions
from robust_rl.robust_rl_tools import load_sketch

import os
import click

import numpy as np
import tensorflow as tf
import random
import math
import time

# RL implementation imports
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.environment.tf_py_environment import TFPyEnvironment
from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.evaluators import evaluate_policy_in_model
from rl_src.tests.general_test_tools import init_args
from rl_src.shielding.shield_processor import ShieldProcessor
import rl_src.shielding.shields
from rl_src.tools.trajectory_buffer import TrajectoryBuffer
from tf_agents.policies.tf_policy import TFPolicy
from rl_src.tools.evaluation_results_class import EvaluationResults
from tf_agents.trajectories import Trajectory

from rl_src.tools.memoryless_fsc_extraction import extract_memory_less_fsc_actions
from rl_src.shielding.shielded_model_checking import model_check_given_policy_and_shield

from tqdm import tqdm

# PAYNT implementation imports
from paynt.parser.sketch import Sketch
from paynt.rl_extension.self_interpretable_interface.black_box_extraction import BlackBoxExtractor

import payntbind
import stormpy

def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    pomdp_sketch = Sketch.load_sketch(
        sketch_path, properties_path)
    return pomdp_sketch


def create_json_file_name(project_path, seed=""):
    """
    Creates a JSON file name based on the project path.
    """
    json_path = os.path.join(project_path, f"benchmark_stats_{seed}.json")
    if os.path.exists(json_path):
        index = 0
        while os.path.exists(os.path.join(project_path, f"benchmark_stats_{seed}_{index}.json")):
            index += 1
        json_path = os.path.join(
            project_path, f"benchmark_stats_{seed}_{index}.json")
    return json_path


def init_extractor(model, args: ArgsEmulator, latent_dim=9, autlearn_extraction=True, steps_to_take=4000, training_epochs=1001) -> BlackBoxExtractor:
    """Function that initializes the FSC extractor/synthesizer.
    Args:
        args (ArgsEmulator): Arguments object containing various settings for the RL and extraction process.
        latent_dim (int, optional): Dimension of the latent space, which defines the maximum size of the FSC provided by SIG. Defaults to 9.
        autlearn_extraction (bool, optional): Selection between SIG extraction and the AALpy Alergia. Defaults to True (Alergia).
        steps_to_take (int, optional): Number of steps, that is taken in each of the parallel simulators. Defaults to 4000.
        training_epochs (int, optional): SIG training epochs irrelevant to Alergia. Defaults to 20001.

    Returns:
        BlackBoxExtractor: Initialized object that performs the SIG or Alergia extraction.
    """
    # family_quotient_numpy = FamilyQuotientNumpy(model)
    direct_extractor = BlackBoxExtractor(memory_len=latent_dim, is_one_hot=True,
                                          use_residual_connection=True, training_epochs=training_epochs,
                                          num_data_steps=steps_to_take, get_best_policy_flag=False,
                                          max_episode_len=args.max_steps,
                                          family_quotient_numpy=None,
                                          autlearn_extraction=autlearn_extraction,
                                          use_gumbel_softmax=True,
                                          non_deterministic=False)
    return direct_extractor


def custom_loop(policy : TFPolicy, environment : EnvironmentWrapperVec, shield_processor : ShieldProcessor, num_parallel_simulations: int, num_steps: int, min_episodes_per_environment: int, trajectory_buffer: TrajectoryBuffer, compile_policy: bool = False, seed: int = None):
    tf_environment = TFPyEnvironment(environment)
    
    # use tf_function for performance if needed -- remove, if the policy is not compatible with TF graph execution
    if compile_policy:
        policy_function = tf.function(policy.distribution)
    else:
        policy_function = policy.distribution
    # Time step is a structure that holds observation (triplet of observation, action mask, integer representing observation index), reward, step type and discount.
    time_step = tf_environment.reset()
    policy_state = policy.get_initial_state(batch_size=num_parallel_simulations)
    prev_actions = tf.zeros((num_parallel_simulations,), dtype=tf.int32)

    for step in range(num_steps*min_episodes_per_environment):
        policy_step = policy_function(time_step, policy_state)
        distribution = policy_step.action
        policy_state = policy_step.state
        # Following operations represents identity, but you can modify it.
        # probs = tf.nn.softmax(distribution.logits).numpy()
        # observation = time_step.observation["observation"].numpy().tolist()
        # mask = time_step.observation["mask"].numpy().tolist()
        # observation_integer = time_step.observation["observation_integer"].numpy().tolist()

        # logits = tf.math.log(probs)
        # End of an identity block.
        if shield_processor is not None:
            logits = shield_processor.compute_new_logits(
                valuations=time_step.observation["observation"].numpy().tolist(),
                integers=time_step.observation["integer"].numpy().tolist(),
                prev_actions=prev_actions.numpy().tolist(),
                played_logits=distribution.logits,
                resets=time_step.is_first().numpy().tolist()
            )
        else:
            logits = distribution.logits

        action = tf.random.categorical(logits, 1, dtype=tf.int32, seed=seed)
        prev_actions = action
        action = tf.reshape(action, (action.shape[0],))
        new_time_step = tf_environment.step(action)
        trajectory = Trajectory(
            step_type=new_time_step.step_type,
            observation=new_time_step.observation,
            action=action,
            policy_info=policy_step.info,
            reward=new_time_step.reward,
            discount=new_time_step.discount,
            next_step_type=new_time_step.step_type
        )
        trajectory_buffer.add_batched_step(trajectory)
        time_step = new_time_step


def set_global_seeds(seed):
    """Set the global random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@click.command()
@click.argument('project', type=click.Path(exists=True))
@click.option("--nu", type=float, default=0.05, help="Safety threshold for the shielding.")
@click.option("--shield", type=click.Choice([None, 'identity', 'standard', 'pessimistic', 'optimistic', 'delta', 'self-constructing-safe', 'self-constructing-unsafe']), default=None, help="Shielding method to use.")
@click.option("--load-agent", type=str, default=None, help="Path to load a pre-trained agent from.")
@click.option("--save-agent", type=str, default="", help="Suffix of folder containing the trained agent.")
@click.option("--agent-training", is_flag=True, default=False, help="Whether to perform agent training.")
@click.option("--deterministic-agent", is_flag=True, default=False, help="Whether the loaded agent is deterministic.")
@click.option("--shield-memory", type=int, default=0, help="Memory size for self-constructing shields. If 0, no memory constraint is applied.")
@click.option("--training-iterations", type=int, default=500, help="Number of iterations for training.")
@click.option("--episode-length", type=int, default=50, help="Maximum length of each episode.")
@click.option("--min-episodes-per-environment", type=int, default=4, help="Minimum number of episodes per parallel environment during evaluation.")
@click.option("--num-environments", type=int, default=512, help="Number of environments to run overall.")
@click.option("--num-parallel-environments", type=int, default=16, help="Number of environments to run in parallel during evaluation.")
@click.option("--model-debug", is_flag=True, default=False, help="Whether to debug the input model.")
@click.option("--save-shield", type=click.Path(), default=None, help="Path to save the shield after evaluation.")
@click.option("--load-shield", type=str, default=None, help="Path to load a pre-trained shield from.")
@click.option("--uniform-random-policy", is_flag=True, default=False, help="Whether to use a uniform random policy for evaluation instead of a trained agent.")
@click.option("--eval-file", type=str, default=None, help="File to save evaluation results.")
@click.option("--model-checking-eval", is_flag=True, default=False, help="Whether to perform model checking based evaluation.")
@click.option("--expected-shield-calls", is_flag=True, default=False, help="Whether to compute expected shield calls and blocked actions during model checking evaluation.")
@click.option("--goal-rew", type=float, default=100.0, help="Reward value for reaching the goal state.")
@click.option("--fail-rew", type=float, default=-100.0, help="Reward value for reaching the fail state.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
def main(project, nu, shield, load_agent, save_agent, agent_training, deterministic_agent, shield_memory, training_iterations, episode_length, min_episodes_per_environment, num_environments, num_parallel_environments, model_debug, save_shield, load_shield, uniform_random_policy, eval_file, model_checking_eval, expected_shield_calls, goal_rew, fail_rew, seed):
    project_path = project
    project_name = os.path.basename(os.path.normpath(project_path))
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     use_rnn_less=True, # Use RNN-less agent (if True, the policy should be completely memoryless)
                     max_steps=episode_length, # Max steps per episode
                     seed=None, # Random seed, for the reproducibility, set it to some integer value
                     prefer_stochastic=True, # Whether to prefer stochastic or deterministic actions during the evaluation
                    )
    set_global_seeds(seed)
    # Replace by your sketch loader.
    sketch = load_sketch(project_path=project_path)

    # ---------------------------------------------------------
    # This is the learning
    model = sketch.pomdp # If you don't have POMDP, you can switch to quotient mdp or some other MDP/POMDP representations.
    # model = sketch.quotient_mdp

    assert "bad" in model.labeling.get_labels(), "Model must have 'bad' label for shielding."

    # TODO investigate this
    # args.batch_size = 1  # For evaluation, we use batch size 1
    args.num_environments = num_environments

    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True, goal_value=goal_rew, antigoal_value=fail_rew)
    
    if save_shield is not None:
        os.makedirs(f"trained_agents/shields/{project_name}", exist_ok=True)
        shield_folder = f"trained_agents/shields/{project_name}/{save_shield}"
    else:
        shield_folder = None
    
    if shield is not None:
        shield_processor = ShieldProcessor(environment.action_keywords, model, nu, shield, args=args, shield_memory=shield_memory, debug=model_debug, shield_folder=shield_folder, deterministic_agent=deterministic_agent)
    else:
        shield_processor = None

    if load_shield is not None:
        if shield is not None:
            print(f"WARNING: Loading shield and therefore ignoring the provided shield type {shield}.")
            shield_processor = None
        shield_processor = ShieldProcessor(environment.action_keywords, model, nu, 'self-constructing-static', args=args, shield_memory=shield_memory, debug=model_debug, shield_folder=None, deterministic_agent=deterministic_agent)
        shield_processor.load_shield(f"trained_agents/shields/{project_name}/{load_shield}")

    if load_agent is not None:
        assert not agent_training, "Cannot load and train an agent at the same time."
        load_folder = f"trained_agents/{project_name}/{load_agent}"
    elif save_agent != "":
        assert agent_training, "Agent folder specified but agent training not enabled."
        load_folder = f"trained_agents/{project_name}/{save_agent}"
    else:
        load_folder = None

    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_agent(
        environment=environment, tf_environment=tf_env, args=args, load=load_folder is not None, agent_folder=load_folder)
    
    if shield_processor:
        environment.set_bad_states(shield_processor.bad_states)

    if agent_training:
        agent.train_agent(iterations=training_iterations)
        print("Training completed.")
        exit()

    if uniform_random_policy:
        print("Using uniform random policy for evaluation.")
        from rl_src.shielding.custom_policy import create_uniform_random_policy
        policy = create_uniform_random_policy(environment)
        compile_policy = False
    else:
        policy = agent.get_policy(False, True)
        policy.set_greedy(False)
        policy.set_policy_masker()
        policy.set_return_real_logits(True)
        # _, obs_to_action = extract_memory_less_fsc_actions(environment, policy, get_probs = True, compile_policy=True)
        # from rl_src.shielding.custom_policy import create_custom_policy
        # policy = create_custom_policy(environment, obs_to_action)
        compile_policy = True

    # Custom simulation loop for evaluation
    if model_checking_eval:
        assert shield_processor is not None, "Shield processor must be provided for model checking evaluation."
        actions, _ = extract_memory_less_fsc_actions(environment, policy, get_probs = not deterministic_agent, compile_policy=compile_policy)

        start_time = time.time()

        mapped_actions = []
        state_choice_labels= []
        for state in range(shield_processor.shield.model_info.model.nr_states):
            obs = shield_processor.shield.model_info.observation_to_state.index(state)
            mapped_distribution, choice_labels = shield_processor.map_played_distribution(actions[obs], state)
            mapped_actions.append(mapped_distribution)
            state_choice_labels.append(choice_labels)

        model_check_result = model_check_given_policy_and_shield(mapped_actions, shield_processor.shield, episode_length=episode_length, goal_value=goal_rew, antigoal_value=fail_rew, expected_shield_calls=expected_shield_calls)      

        eval_elapsed_time = time.time() - start_time
        print(f"Model checking evaluation took {eval_elapsed_time:.2f} seconds.")  

        print(f"{model_check_result['full_safety_probability']};{model_check_result['actual_reward']};{model_check_result['expected_shield_calls']};{model_check_result['expected_blocked_actions']};{model_check_result['earliest_shielded_step']};{eval_elapsed_time}")

        if eval_file is not None:
            file_exists = os.path.exists(eval_file)
            with open(eval_file, "a") as f:
                if not file_exists:
                    f.write("project_name;agent;shield;shield_memory;nu;safety_probability;full_safety_probability;goal_reachability;reward;shield_calls;blocked_actions;evaluation_time\n")
                if uniform_random_policy:
                    agent_str = "uniform_random"
                else:
                    agent_str = load_agent
                if load_shield is not None:
                    shield = f"constructed-{shield}"
                f.write(f"{project_name};{agent_str};{shield};{shield_memory};{nu};") 
                f.write(f'{model_check_result["safety_probability"]};{model_check_result["full_safety_probability"]};{model_check_result["goal_reachability"]};{model_check_result["actual_reward"]};{model_check_result["expected_shield_calls"]};{model_check_result["expected_blocked_actions"]};{eval_elapsed_time}')
                f.write("\n")

        exit()
    else:
        trajectory_buffer = TrajectoryBuffer(environment, truncation_point=episode_length)
        evaluation_result = EvaluationResults()
        num_eval_iterations = math.ceil(num_environments / num_parallel_environments)
        environment.temporarily_set_num_envs(num_parallel_environments)
        start_time = time.time()
        for eval_iteration in tqdm(range(num_eval_iterations)):
            custom_loop(policy, environment, shield_processor, num_parallel_environments, episode_length, min_episodes_per_environment, trajectory_buffer, compile_policy=compile_policy, seed=seed)
            if save_shield is not None and shield_processor is not None:
                shield_processor.save_shield(shield_processor.shield_folder, iteration=eval_iteration)
            trajectory_buffer.final_update_of_results(evaluation_result.update)
            trajectory_buffer.clear()
        eval_elapsed_time = time.time() - start_time
        environment.reset_num_envs() # Sets the number of environments back to original value.
        print(f"Evaluation loop took {eval_elapsed_time:.2f} seconds.")
        # trajectory_buffer.final_update_of_results(evaluation_result.update, truncate=False)
        # evaluation_result.log_evaluation_info()

        eval_result = evaluation_result.compute_weighted_evaluation_info()
        print(eval_result)
        print(f'Episodes: {eval_result["counted_episodes"]}\nAverage episode length: {eval_result["average_episode_length"]}\nReward: {eval_result["virtual_returns"]}\n Goal reach probabilities: {eval_result["reach_probs"]}\nBad outcome prob: {eval_result["average_bad_outcome_prob"]}\nExpected allowed actions: {1 - (shield_processor.shield.blocked_actions / shield_processor.shield.shield_calls) if shield_processor.shield.shield_calls > 0 else 0}')
        # exit()

    # Simulator
    # evaluation_result = evaluate_policy_in_model(policy, args, environment, tf_env, max_steps=episode_length, shield_processor=shield_processor)

    # ---------------------------------------------------------

    # Save the results. Now the results are stored in the same folder as the processed models, but you can change it as needed.
    # json_path = create_json_file_name(project_path, seed=args.seed)
    # agent.evaluation_result.save_to_json(json_path, new_pomdp=False)

    if shield_processor:
        if type(shield_processor.shield) in [rl_src.shielding.shields.SelfConstructingShieldConstructionSafe]:
            shield_processor.shield.finalize_all_unfinished_traces()
        if shield_processor.shield_folder is not None:
            shield_processor.save_shield(shield_processor.shield_folder, "final")
        print()
        print("Shield stats:")
        print(f"Shield calls: {shield_processor.shield.shield_calls}")
        print(f"Blocked actions: {shield_processor.shield.blocked_actions}")
        # print(f"Bad episodes encountered during evaluation: {shield_processor.bad_epsisodes} ({shield_processor.bad_epsisodes / evaluation_result.counted_episodes[-1]})")
        if type(shield_processor.shield) in [rl_src.shielding.shields.SelfConstructingShield, rl_src.shielding.shields.SelfConstructingShieldConstructionSafe, rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe]:
            if shield_processor.shield.memory > 0:
                final_allow_mdp = payntbind.synthesis.createMdpFromVectorMatrix(shield_processor.shield.memory_unfolded_model, shield_processor.shield.current_matrix_vector)
                result = stormpy.model_checking(final_allow_mdp, shield_processor.shield.safety_property[0])
                print(f"Initial state value of allow MDP: {result.get_values()[final_allow_mdp.initial_states[0]]}")
            else:
                print(f"Tree size: {shield_processor.shield.initial_node.number_of_tree_nodes()}")
                print(f"Initial node values: {shield_processor.shield.initial_node.value}")
            print(f"Added non-optimal actions: {shield_processor.shield.added_nonoptimal_actions}")
            print(f"Backpropagation calls: {shield_processor.shield.backpropagation_calls}")
            print(f"Model checking calls: {shield_processor.shield.model_checking_calls}")

        print()
        print(eval_result["counted_episodes"], eval_result["average_episode_length"], shield_processor.shield.shield_calls, eval_elapsed_time, eval_result["virtual_returns"], eval_result["reach_probs"],  eval_result["average_bad_outcome_prob"], shield_processor.shield.blocked_actions, shield_processor.shield.added_nonoptimal_actions, end=";", sep=";")
        if type(shield_processor.shield) in [rl_src.shielding.shields.SelfConstructingShield, rl_src.shielding.shields.SelfConstructingShieldConstructionSafe, rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe]:
            if shield_processor.shield.memory > 0:
                print(result.get_values()[final_allow_mdp.initial_states[0]], end="", sep=";")
            else:
                print(shield_processor.shield.initial_node.value, shield_processor.shield.initial_node.number_of_tree_nodes(), end="", sep=";")
        print()

        if eval_file is not None:
            file_exists = os.path.exists(eval_file)
            with open(eval_file, "a") as f:
                if not file_exists:
                    f.write("project_name;agent;shield;shield_memory;nu;counted_episodes;average_episode_length;shield_calls;eval_elapsed_time;reward;goal_probability;bad_outcome_prob;blocked_actions;added_nonoptimal_actions;initial_node_value;tree_size\n")
                if uniform_random_policy:
                    agent_str = "uniform_random"
                else:
                    agent_str = load_agent
                if load_shield is not None:
                    shield = f"constructed-{shield}"
                f.write(f"{project_name};{agent_str};{shield};{shield_memory};{nu};") 
                f.write(f'{eval_result["counted_episodes"]};{eval_result["average_episode_length"]};{shield_processor.shield.shield_calls};{eval_elapsed_time};{eval_result["virtual_returns"]};{eval_result["reach_probs"]};{eval_result["average_bad_outcome_prob"]};{shield_processor.shield.blocked_actions};{shield_processor.shield.added_nonoptimal_actions};')
                if type(shield_processor.shield) in [rl_src.shielding.shields.SelfConstructingShield, rl_src.shielding.shields.SelfConstructingShieldConstructionSafe, rl_src.shielding.shields.SelfConstructingShieldConstructionUnsafe]:
                    if shield_processor.shield.memory > 0:
                        f.write(f"{result.get_values()[final_allow_mdp.initial_states[0]]};\n")
                    else:
                        f.write(f"{shield_processor.shield.initial_node.value};{shield_processor.shield.initial_node.number_of_tree_nodes()}\n")
                else:
                    f.write(";\n")

        print(f"{eval_result['average_bad_outcome_prob']};{eval_result['virtual_returns']};{shield_processor.shield.shield_calls};{shield_processor.shield.blocked_actions};None;{eval_elapsed_time}")


if __name__ == "__main__":
    main()
