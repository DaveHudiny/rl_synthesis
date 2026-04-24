import os
import click
import subprocess
import tqdm
import itertools
import pandas as pd


@click.command()
@click.option("--results-folder", type=str, required=True, help="Path to the results folder.")
@click.option("--shield-memory", type=int, required=False, default=0, help="Memory size for the shield.")
@click.option("--shield-name", type=str, required=False, default="", help="Path to the shield file if applicable.")
@click.option("--number-of-evaluations", type=int, required=False, default=1, help="Number of evaluations to run for the shield.")
@click.option("--shield-nu", type=float, required=False, default=None, help="Nu parameter for shielding.")
@click.option("--shield-model", type=str, required=False, default=None, help="Model to use for evaluation: 'dpm', 'corridor', or 'drone'.")
@click.option("--shield-construction-agent", type=str, required=False, default=None, help="Agent used for shield construction if applicable.")
@click.option("--artifact-review", is_flag=True, default=False, help="Whether to only generate artifact review data.")
@click.option("--smoke-test", is_flag=True, default=False, help="Whether to only generate smoke test data.")
def main(results_folder, shield_memory, shield_name, number_of_evaluations, shield_nu, shield_model, shield_construction_agent, artifact_review, smoke_test):

    # init results file
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)
    script_raw_results_path = os.path.join(results_folder, "construction_raw_evaluation_results.csv")
    script_log_path = os.path.join(results_folder, "construction_evaluation_log.log")

    # init eval parameters
    models = {'dpm' : 'models/shielding/dpm', 'corridor' : 'models/shielding/test-corridor', 'drone' : 'models/shielding/slippy-drone', 'drone-b' : 'models/shielding/collect'} # USED FOR GENERATING COMMAND
    agents = {'dpm' : {'greedy' : 'greedy-iter-1000 --deterministic-agent', 'safe' : 'safe-iter-100 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'corridor' : {'greedy' : 'greedy-iter-100 --deterministic-agent', 'safe' : 'safe-iter-4000 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'drone' : {'greedy' : 'greedy-iter-4000 --deterministic-agent', 'safe' : 'safe-iter-1000 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'drone-b' : {'greedy' : 'greedy-iter-500 --deterministic-agent', 'safe' : 'safe-iter-500', 'random' : 'random --uniform-random-policy'}} # USED FOR GENERATING COMMAND
    nus = {'dpm' : [0.01, 0.05, 0.2], 'corridor' : [0.05, 0.1, 0.2], 'drone' : [0.01, 0.05, 0.2], 'drone-b' : [0.01, 0.05, 0.2]} # USED FOR GENERATING COMMAND
    model_settings = {'dpm' : '--goal-rew 0.0 --fail-rew 0.0', 'corridor' : '', 'drone' : '', 'drone-b' : ''} # USED FOR GENERATING COMMAND
    environments_settings = "--num-environments 2048 --num-parallel-environments 256"
    if artifact_review:
        environments_settings = "--num-environments 512 --num-parallel-environments 256"
    elif smoke_test:
        environments_settings = "--num-environments 64 --num-parallel-environments 32"

    assert shield_model is None or shield_model in models.keys(), "If specified, shield model must be one of: 'dpm', 'corridor', 'drone'"
    assert shield_construction_agent is None or shield_construction_agent in ['greedy', 'safe', 'random'], "If specified, shield construction agent must be one of: 'greedy', 'safe', 'random'"

    # prepare what to run combinations
    eval_combinations = []
    model_list = models.keys() if not shield_model else [shield_model]
    for model in model_list:
        agent_list = agents[model].keys() if not shield_construction_agent else [shield_construction_agent]
        for agent in agent_list:
            if shield_nu is not None:
                nu_list = [nu for nu in nus[model] if nu == shield_nu]
            else:
                nu_list = nus[model]
            for nu in nu_list:
                for eval_idx in range(number_of_evaluations):
                    eval_combinations.append( (model, agent, nu, eval_idx) )

    for model, agent, nu, eval_idx in tqdm.tqdm(eval_combinations):

        project_name = os.path.basename(os.path.normpath(models[model]))

        save_shield_name = f"{agent}-nu{str(nu).replace('.','')}-mem{shield_memory}-{shield_name}-eval-{eval_idx}"

        if smoke_test:
            save_shield_name = f"{agent}-nu{str(nu).replace('.','')}-mem{shield_memory}-smoke-test-eval-{eval_idx}"

        full_shield_path = 'trained_agents/shields/' + project_name + '/' + save_shield_name + '-iter-final-shield' + '.pickle'

        if os.path.exists(full_shield_path):
            print(f"Shield {full_shield_path} already exists. Skipping shield construction.")
            continue

        command = f"python3 shielding.py {models[model]} --episode-length 50 --load-agent {agents[model][agent]} --shield self-constructing-unsafe --shield-memory {shield_memory} --nu {nu} {model_settings[model]} --eval-file {script_raw_results_path} {environments_settings} --save-shield {save_shield_name}"

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output, _ = process.communicate()
        with open(script_log_path, "a") as f:
            f.write(output.decode())

if __name__ == "__main__":
    main()
