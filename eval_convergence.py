import os
import click
import subprocess
import tqdm
import itertools
import pandas as pd

def add_result_to_csv(csv_path, df, model, agent, nu, shield, shield_name, iter, risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_time):
    exists = (
            (df['model'] == model)
            & (df['agent'] == agent)
            & (df['nu'] == nu)
            & (df['shield'] == shield)
            & (df['shield_name'] == shield_name)
            & (df['iter'] == iter)
        ).any()
    if exists:
        print(f"Result for model={model}, agent={agent}, nu={nu}, shield={shield}, shield_name={shield_name}, iter={iter} already exists in CSV. Skipping addition.")
        return
    with open(csv_path, "a") as f:
        f.write(f"{model},{agent},{nu},{shield},{shield_name},{iter},{risk},{reward},{shield_calls},{blocked_actions},{earliest_shielded_step},{eval_time}\n")


@click.command()
@click.option("--results-folder", type=str, required=True, help="Path to the results folder.")
@click.option("--shield-memory", type=int, required=False, default=0, help="Memory size for the shield.")
@click.option("--shield-nu", type=float, required=False, default=None, help="Nu parameter for shielding.")
@click.option("--shield-model", type=str, required=False, default=None, help="Model to use for evaluation: 'dpm', 'corridor', or 'drone'.")
@click.option("--shield-construction-agent", type=str, required=False, default=None, help="Agent used for shield construction if applicable.")
@click.option("--shield-name", type=str, required=False, default="", help="Name of the shield to use.")
@click.option("--construct-shields", is_flag=True, default=False, help="Only run self-constructing shields.")
@click.option("--number-of-evaluations", type=int, required=False, default=3, help="Number of evaluations to run for the shield.")
def main(results_folder, shield_memory, shield_nu, shield_model, shield_construction_agent, shield_name, construct_shields, number_of_evaluations):

    # init results file
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)
    main_csv_path = os.path.join(results_folder, "convergence.csv")
    header = "model,agent,nu,shield,shield_name,iter,risk,reward,shield_calls,blocked_actions,earliest_shielded_step,eval_time\n"
    if not os.path.exists(main_csv_path):
        with open(main_csv_path, "w") as f:
            f.write(header)
    else:
        # Check header
        with open(main_csv_path, "r") as f:
            first_line = f.readline()
        if first_line != header:
            raise RuntimeError(f"Header mismatch in {main_csv_path}. Expected: {header.strip()} Found: {first_line.strip()}")
        print(f"Results file {main_csv_path} already exists. Appending new results.")

    # Parse the CSV data into a DataFrame
    df = pd.read_csv(main_csv_path)

    constructed_shields = []
    script_raw_results_path = os.path.join(results_folder, "raw_convergence_results.csv")
    script_log_path = os.path.join(results_folder, "convergence_log.log")
    if construct_shields:
        script_raw_results_path = os.path.join(results_folder, "raw_shield_construction_results.csv")
        script_log_path = os.path.join(results_folder, "shield_construction_log.log")
        constructed_shields_list_path = os.path.join(results_folder, "constructed_shields_list.txt")
        # Parse constructed shields list if it exists
        if os.path.exists(constructed_shields_list_path):
            with open(constructed_shields_list_path, "r") as f:
                constructed_shields = [line.strip() for line in f if line.strip()]
    log_settings = f"--eval-file {script_raw_results_path}" # USED FOR GENERATING COMMAND


    # init eval parameters
    models = {'dpm' : 'models/shielding/dpm', 'corridor' : 'models/shielding/test-corridor', 'drone' : 'models/shielding/slippy-drone', 'drone-b' : 'models/shielding/collect'} # USED FOR GENERATING COMMAND
    agents = {'dpm' : {'greedy' : 'greedy-iter-1000 --deterministic-agent', 'safe' : 'safe-iter-100 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'corridor' : {'greedy' : 'greedy-iter-100 --deterministic-agent', 'safe' : 'safe-iter-4000 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'drone' : {'greedy' : 'greedy-iter-4000 --deterministic-agent', 'safe' : 'safe-iter-1000 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'drone-b' : {'rabdom' : 'random -- uniform-random-policy'}} # USED FOR GENERATING COMMAND
    nus = {'dpm' : [0.01, 0.05, 0.2], 'corridor' : [0.05, 0.1, 0.2], 'drone' : [0.01, 0.05, 0.2], 'drone-b' : [0.2]} # USED FOR GENERATING COMMAND
    model_settings = {'dpm' : '--goal-rew 0.0 --fail-rew 0.0', 'corridor' : '', 'drone' : '', 'drone-b' : ''} # USED FOR GENERATING COMMAND

    shield_string = f"--shield self-constructing-unsafe --shield-memory {shield_memory}"
    EVAL_NUMBER = 0

    # prepare what to run combinations
    eval_combinations = []
    model_list = models.keys() if shield_model is None else [shield_model]
    for model in model_list:
        agent_list = agents[model].keys() if shield_construction_agent is None else [shield_construction_agent]
        assert all(agent in agents[model].keys() for agent in agent_list), f"One or more specified eval agents are not valid for model {model}."
        for agent in agent_list:
            nu_list = nus[model] if shield_nu is None else [shield_nu]
            assert all(nu in nus[model] for nu in nu_list), f"One or more specified nus are not valid for model {model}."
            for nu in nu_list:
                eval_combinations.append( (model, agent, nu) )


    for model, agent, nu in tqdm.tqdm(eval_combinations):
        
        shield_partial_name = f"{agent}-mem_{shield_memory}-nu_{str(nu).replace('.','')}-{shield_name}-eval_{EVAL_NUMBER}"

        if not construct_shields:

            for current_iter in range(0, 256, 4):

                # Check if result already exists in df
                exists = (
                    (df['model'] == model)
                    & (df['agent'] == agent)
                    & (df['nu'] == nu)
                    & (df['shield'] == "constructed")
                    & (df['shield_name'] == shield_partial_name)
                    & (df['iter'] == current_iter)
                ).any()

                if exists:
                    print(f"Result for model={model}, agent={agent}, nu={nu}, shield=constructed, shield_name={shield_partial_name}, iter={current_iter} already exists in CSV. Skipping evaluation.")
                    continue

                command = f"python3 shielding.py {models[model]} --episode-length 50 --load-agent {agents[model][agent]} {shield_string} --load-shield {shield_partial_name}-iter-{current_iter}-shield.pickle --nu {nu} {model_settings[model]} {log_settings} --model-checking-eval --expected-shield-calls"

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

                # Process the last line of the output
                last_line = output.decode().strip().split('\n')[-1]
                if ';' not in last_line:
                    print("Error: Output does not contain ';' in the last line.")
                else:
                    risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_time = [part.strip() for part in last_line.split(';', 5)]

                    add_result_to_csv(main_csv_path, df, model, agent, nu, "constructed", shield_partial_name, current_iter, risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_time)

        else:

            for eval_it in range(number_of_evaluations):

                shield_partial_name = f"{agent}-mem_{shield_memory}-nu_{str(nu).replace('.','')}-{shield_name}-eval_{eval_it}"

                model_shield_partial_name = f"{model}-{shield_partial_name}"

                if model_shield_partial_name in constructed_shields:
                    print(f"Shield {model_shield_partial_name} already constructed. Skipping construction.")
                    continue

                command = f"python3 shielding.py {models[model]} --episode-length 50 --load-agent {agents[model][agent]} {shield_string} --save-shield {shield_partial_name} --nu {nu} {model_settings[model]} {log_settings} --num-environments 2048 --num-parallel-environments 8 --min-episodes-per-environment 10"

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
 
                # Mark shield as constructed
                with open(constructed_shields_list_path, "a") as f:
                    f.write(f"{model_shield_partial_name}\n")


if __name__ == "__main__":
    main()