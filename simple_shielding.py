from rl_src.shielding.shielding_options import ShieldingOptions
from robust_rl.robust_rl_tools import load_sketch

import os
import click

import numpy as np
import tensorflow as tf
import random

# RL implementation imports
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.environment.tf_py_environment import TFPyEnvironment
from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.evaluators import evaluate_policy_in_model
from rl_src.tests.general_test_tools import init_args
from rl_src.shielding.shield_processor import ShieldProcessor
import rl_src.shielding.shields

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

def set_global_seeds(seed):
    """Set the global random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@click.command()
@click.argument('project', type=click.Path(exists=True))
@click.option("--nu", type=float, default=0.05, help="Safety threshold for the shielding.")
@click.option("--shield", type=click.Choice([None, 'identity', 'standard', 'pessimistic', 'optimistic', 'delta', 'self-constructing', 'self-constructing-unsafe']), default=None, help="Shielding method to use.")
@click.option("--agent-folder", type=str, default="", help="Suffix of folder containing the trained agent.")
@click.option("--agent-training", is_flag=True, default=False, help="Whether to perform agent training.")
@click.option("--shield-memory", type=int, default=0, help="Memory size for self-constructing shields. If 0, no memory constraint is applied.")
def main(project, nu, shield, agent_folder, agent_training, shield_memory):
    project_path = project
    project_name = os.path.basename(os.path.normpath(project_path))
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     use_rnn_less=False, # Use RNN-less agent (if True, the policy should be completely memoryless)
                     max_steps=50, # Max steps per episode
                     seed=None, # Random seed, for the reproducibility, set it to some integer value
                     prefer_stochastic=True, # Whether to prefer stochastic or deterministic actions during the evaluation
                    )
    set_global_seeds(args.seed)
    # Replace by your sketch loader.
    sketch = load_sketch(project_path=project_path)

    # ---------------------------------------------------------
    # This is the learning
    model = sketch.pomdp # If you don't have POMDP, you can switch to quotient mdp or some other MDP/POMDP representations.
    # model = sketch.quotient_mdp

    # TODO investigate this
    # args.batch_size = 1  # For evaluation, we use batch size 1
    args.num_environments = 512

    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True)
    
    if shield is not None:
        shield_processor = ShieldProcessor(environment.action_keywords, model, nu, shield, args=args, shield_memory=shield_memory) # Placeholder for your implementation.
    else:
        shield_processor = None

    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_agent(
        environment=environment, tf_environment=tf_env, args=args, load=True, agent_folder=f"trained_agents/{project_name}-{agent_folder}")
    
    if agent_training:
        agent.train_agent(iterations=300)

    policy = agent.get_policy(False, True)
    policy.set_greedy(False)
    policy.set_policy_masker()
    policy.set_return_real_logits(True)
    evaluate_policy_in_model(policy, args, environment, tf_env, max_steps=100, shield_processor=shield_processor)
    # ---------------------------------------------------------

    # Save the results. Now the results are stored in the same folder as the processed models, but you can change it as needed.
    # json_path = create_json_file_name(project_path, seed=args.seed)
    # agent.evaluation_result.save_to_json(json_path, new_pomdp=False)

    if shield_processor:
        print()
        print("Shield stats:")
        print(f"Shield calls: {shield_processor.shield.shield_calls}")
        print(f"Blocked actions: {shield_processor.shield.blocked_actions}")
        if type(shield_processor.shield) in [rl_src.shielding.shields.SelfConstructingShield, rl_src.shielding.shields.SelfConstructingShieldUnsafe]:
            if shield_processor.shield.memory > 0:
                final_allow_mdp = payntbind.synthesis.createMdpFromVectorMatrix(shield_processor.shield.memory_unfolded_model, shield_processor.shield.current_matrix_vector)
                result = stormpy.model_checking(final_allow_mdp, shield_processor.shield.safety_property[0])
                print(f"Initial state value of allow MDP: {result.get_values()[final_allow_mdp.initial_states[0]]}")
            else:
                for node in shield_processor.shield.current_nodes:
                    shield_processor.shield.back_propagate_values(node)
                print(f"Tree size: {shield_processor.shield.initial_node.number_of_tree_nodes()}")
                print(f"Initial node values: {shield_processor.shield.initial_node.value}")
            print(f"Added non-optimal actions: {shield_processor.shield.added_nonoptimal_actions}")


if __name__ == "__main__":
    main()
