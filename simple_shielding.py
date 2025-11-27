from shielding.shielding_options import ShieldingOptions
from robust_rl.robust_rl_tools import load_sketch

import os

import numpy as np
import tensorflow as tf
import random

# RL implementation imports
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment.tf_py_environment import TFPyEnvironment
from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from tools.args_emulator import ArgsEmulator
from tools.evaluators import evaluate_policy_in_model
from tests.general_test_tools import init_args
from shielding.shield_processor import ShieldProcessor

# PAYNT implementation imports
from paynt.parser.sketch import Sketch
from paynt.rl_extension.self_interpretable_interface.black_box_extraction import BlackBoxExtractor


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

def main():
    project_path = "models/models_pomdp_no_family/network-3-8-20"
    # project_path = "mdp_obstacles/"
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     use_rnn_less=False, # Use RNN-less agent (if True, the policy should be completely memoryless)
                     max_steps=601, # Max steps per episode
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

    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True)
    
    shield_processor = ShieldProcessor(len(environment.action_keywords), args=args) # Placeholder for your implementation.

    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_agent(
        environment=environment, tf_environment=tf_env, args=args, load=False, agent_folder="trained_agents")
    # agent.train_agent(iterations=500)
    agent.train_agent_shielded(iterations=500, shield_processor=shield_processor,
                              shielded_option=ShieldingOptions.NAIVE_TRAINING_SHIELDING)
    policy = agent.get_policy(False, True)
    policy.set_greedy(False)
    policy.set_policy_masker()
    policy.set_return_real_logits(True)
    evaluate_policy_in_model(policy, args, environment, tf_env, shield_processor=shield_processor)
    # ---------------------------------------------------------

    # Save the results. Now the results are stored in the same folder as the processed models, but you can change it as needed.
    json_path = create_json_file_name(project_path, seed=args.seed)
    agent.evaluation_result.save_to_json(json_path, new_pomdp=False)


if __name__ == "__main__":
    main()
