from robust_rl.robust_rl_tools import load_sketch

import os

import numpy as np

# RL implementation imports
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment.tf_py_environment import TFPyEnvironment
from agents.recurrent_ppo_agent import Recurrent_PPO_Agent
from agents.recurrent_sac_agent import Recurrent_SAC_Agent
from agents.father_agent import FatherAgent
from interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from tools.args_emulator import ArgsEmulator, ReplayBufferOptions, EvaluationOptions
from tools.evaluators import evaluate_policy_in_model
from tools.evaluation_results_class import EvaluationResults
from tests.general_test_tools import init_args
from agents.policies.policy_mask_wrapper import PolicyMaskWrapper

# PAYNT implementation imports
from paynt.parser.sketch import Sketch
from paynt.rl_extension.self_interpretable_interface.black_box_extraction import BlackBoxExtractor
from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC
from paynt.quotient.fsc import FscFactored

from robust_rl.robust_rl_tools import set_global_seeds

class EvaluationOptionResult:
    def __init__(self, seed):
        self.seed = seed
        self.returns_per_evaluation_option = {}
        self.reachabilities_per_evaluation_option = {}

    def add_option_result(self, evaluation_option: EvaluationOptions, returns: list[float], reachabilities: list[float]):
        self.returns_per_evaluation_option[evaluation_option.name] = returns
        self.reachabilities_per_evaluation_option[evaluation_option.name] = reachabilities

    def save_to_json(self, json_path):
        import json
        data = {
            "seed": self.seed,
            "returns_per_evaluation_option": {option: str(self.returns_per_evaluation_option[option]) for option in self.returns_per_evaluation_option},
            "reachabilities_per_evaluation_option": {option: str(self.reachabilities_per_evaluation_option[option]) for option in self.reachabilities_per_evaluation_option},
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)


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


def fsc_extraction(model, agent: FatherAgent) -> tuple[FscFactored, TableBasedPolicy]:

    direct_extractor = init_extractor(
        model, agent.args, autlearn_extraction=True)
    policy = agent.get_policy(False, True)
    policy.set_greedy(True)  # Ensures, that the agent selects argmax actions
    # Ensures, that the agent respects the action masking during extraction
    policy.set_policy_masker()
    extracted_fsc, extraction_stats = direct_extractor.clone_and_generate_fsc_from_policy(  # Extraction stats is a structure that contains various statistics about the extraction process
        policy, agent.environment, agent.tf_environment)
    # We can evaluate the extracted FSC in the model to see its performance
    evaluate_policy_in_model(extracted_fsc, agent.args,
                             agent.environment, agent.tf_environment)

    # Generate deterministic PAYNT FSC representation.
    paynt_fsc = ConstructorFSC.construct_fsc_from_table_based_policy(
        extracted_fsc, pomdp_quotient=model, family_quotient_numpy=None, cut_probs=1.0)
    tf_fsc_policy = TableBasedPolicy(original_policy=policy,  # TODO: Replace the original policy from the policy initialization process.
                                     # If you provide the action and update functions (shape [memory_size, nr_observations]), you obtain standard TF Policy that can be evaluated in TF-Agents environment.
                                     action_function=np.array(
                                         paynt_fsc.action_function),
                                     update_function=np.array(
                                         paynt_fsc.update_function),
                                     action_keywords=paynt_fsc.action_labels
                                     )
    evaluate_policy_in_model(tf_fsc_policy, agent.args,
                             agent.environment, agent.tf_environment)

    policy.set_greedy(False)  # Reset the policy to non-greedy mode
    # Unset the policy masker to allow all actions again, TODO: Check if this is necessary
    policy.set_identity_masker()
    return paynt_fsc, tf_fsc_policy


def evaluate_in_all_options(policy: PolicyMaskWrapper, args: ArgsEmulator, environment: EnvironmentWrapperVec, tf_env: TFPyEnvironment, evaluation_results: dict[int, EvaluationResults] = None):
    policy.set_policy_masker()
    for eval_option in EvaluationOptions:
        if eval_option == EvaluationOptions.FULL_STOCHASTIC:
            policy.set_random_selector()
        elif eval_option == EvaluationOptions.ARGMAX:
            policy.set_argmax_selector()
        elif eval_option == EvaluationOptions.PRUNING_0DOT05:
            policy.set_prune_zero_dot_zero_five_probs_selector()
        elif eval_option == EvaluationOptions.PRUNING_0DOT1:
            policy.set_prune_zero_dot_one_probs_selector()
        elif eval_option == EvaluationOptions.UNIFORM_PRUNING_0DOT1:
            policy.set_prune_below_zero_dot_one_uniform_selector()
        elif eval_option == EvaluationOptions.UNIFORM_TOP_THREE:
            policy.set_uniform_top_three_selector()
        elif eval_option == EvaluationOptions.UNIFORM_TOP_TWO:
            policy.set_uniform_top_two_selector()
        else:
            raise ValueError(f"Unknown evaluation option: {eval_option}")
        if eval_option not in evaluation_results:
            evaluation_results[eval_option] = None
        evaluation_results[eval_option] = evaluate_policy_in_model(
            policy,
            args,
            environment,
            tf_env,
            evaluation_result=evaluation_results[eval_option]
        )
    policy.set_identity_masker()  # Reset the masker after evaluation
    policy.set_random_selector()  # Reset to random selector for next training


def evaluate_single_pomdp(project_path: str, training_iterations_per_evaluation=100, meta_iterations=40, seed=None, evaluation_option=None) -> EvaluationOptionResult:
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     use_rnn_less=False,
                     max_steps=601,
                     seed=seed,
                     prefer_stochastic=True)

    model = load_sketch(project_path=project_path).pomdp
    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True)
    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_Agent(
        environment=environment, tf_environment=tf_env, args=args, load=False)
    evaluation_results = {}
    policy = agent.get_policy(False, True)
    evaluate_in_all_options(policy, args, environment, tf_env, evaluation_results)
    for meta_iteration in range(meta_iterations):
        agent.train_agent(iterations=training_iterations_per_evaluation)
        policy = agent.get_policy(False, True)
        evaluate_in_all_options(policy, args, environment, tf_env, evaluation_results)

    evaluation_option_result = EvaluationOptionResult(seed=seed)
    for eval_option in evaluation_results:
        evaluation_option_result.add_option_result(
            eval_option,
            evaluation_results[eval_option].returns,
            evaluation_results[eval_option].reach_probs
        )

    return evaluation_option_result


def run_evaluation_experiment(training_iterations_per_evaluation=100, meta_iterations=40, project_folder="models/models_pomdp_no_family/"):
    import os
    import sys
    available_models = [f for f in os.listdir(
        project_folder) if os.path.isdir(os.path.join(project_folder, f))]
    for seed in [12345, 23456, 34567, 45678, 56789]:
        for model_name in available_models:
            print(f"Evaluating model {model_name} with seed {seed}")
            evaluation_result = evaluate_single_pomdp(os.path.join(project_folder, model_name),
                                                      training_iterations_per_evaluation,
                                                      meta_iterations,
                                                      seed=seed)
            json_path = create_json_file_name(
                os.path.join(project_folder, model_name), seed=seed)
            evaluation_result.save_to_json(json_path)


def main():
    project_path = "models/models_pomdp_no_family/refuel-10"
    # project_path = "mdp_obstacles/"
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     use_rnn_less=False, # Use RNN-less agent (if True, the policy should be completely memoryless)
                     max_steps=601, # Max steps per episode
                     seed=42, # Random seed, for the reproducibility, set it to some integer value
                     prefer_stochastic=True, # Whether to prefer stochastic or deterministic actions during the evaluation
                     stochastic_environment_actions=True,
                     masked_training=False
                    )
    # Replace by your sketch loader.
    set_global_seeds(args.seed)
    sketch = load_sketch(project_path=project_path)

    # ---------------------------------------------------------
    # This is the learning
    # If you don't have POMDP, you can switch to quotient mdp or some other MDP/POMDP representations.
    model = sketch.pomdp
    # model = sketch.quotient_mdp

    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True)

    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_Agent(
        environment=environment, tf_environment=tf_env, args=args, load=False, agent_folder="trained_agents")
    agent.train_agent(iterations=2000)
    policy = agent.get_policy(False, True)
    evaluate_policy_in_model(policy, args, environment, tf_env)
    json_path = create_json_file_name(project_path, seed=args.seed)
    agent.evaluation_result.save_to_json(json_path, new_pomdp=False)
    # ---------------------------------------------------------

    # This performs the extraction.

    # if hasattr(sketch, 'pomdp'): # Ensure, that the sketch has a POMDP representation
    #     paynt_fsc, tf_fsc = fsc_extraction(sketch, agent)

    # Save the results. Now the results are stored in the same folder as the processed models, but you can change it as needed.


if __name__ == "__main__":
    main()
