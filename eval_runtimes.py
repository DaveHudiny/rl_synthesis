import os
import click
import subprocess
import tqdm
import itertools
import pandas as pd


def add_result_to_csv(csv_path, df, model, agent, nu, shield, shield_name, risk, reward, shield_calls, blocked_actions, earliest_shielded_step,eval_time):
    exists = (
            (df['model'] == model)
            & (df['agent'] == agent)
            & (df['nu'] == nu)
            & (df['shield'] == shield)
            & (df['shield_name'] == shield_name)
        ).any()
    if exists:
        print(f"Result for model={model}, agent={agent}, nu={nu}, shield={shield}, shield_name={shield_name} already exists in CSV. Skipping addition.")
        return
    with open(csv_path, "a") as f:
        f.write(f"{model},{agent},{nu},{shield},{shield_name},{risk},{reward},{shield_calls},{blocked_actions},{earliest_shielded_step},{eval_time}\n")


@click.command()
@click.option("--results-folder", type=str, required=True, help="Path to the results folder.")
@click.option("--shield-memory", type=int, required=False, default=0, help="Memory size for the shield.")
@click.option("--shield-path", type=str, required=False, default="greedy-mem_0-nu_02--eval_0-iter-final-shield.pickle", help="Path to the shield file if applicable.")
@click.option("--number-of-evaluations", type=int, required=False, default=3, help="Number of evaluations to run for the shield.")
def main(results_folder, shield_memory, shield_path, number_of_evaluations):

    # init results file
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)
    main_csv_path = os.path.join(results_folder, "evaluation_results.csv")
    header = "model,agent,nu,shield,shield_name,risk,reward,shield_calls,blocked_actions,earliest_shielded_step,eval_time\n"
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

    script_raw_results_path = os.path.join(results_folder, "raw_runtime_evaluation_results.csv")
    script_log_path = os.path.join(results_folder, "runtime_evaluation_log.log")
    log_settings = f"--eval-file {script_raw_results_path}" # USED FOR GENERATING COMMAND

    # init eval parameters
    models = {'dpm' : 'models/shielding/dpm', 'corridor' : 'models/shielding/test-corridor', 'drone' : 'models/shielding/slippy-drone', 'drone-b' : 'models/shielding/collect'} # USED FOR GENERATING COMMAND
    agents = {'dpm' : {'greedy' : 'greedy-iter-1000 --deterministic-agent'},
              'corridor' : {'greedy' : 'greedy-iter-100 --deterministic-agent'},
              'drone' : {'greedy' : 'greedy-iter-4000 --deterministic-agent'},
              'drone-b' : {'greedy' : 'random --uniform-random-policy'}} # USED FOR GENERATING COMMAND
    # agents = {'dpm' : {'greedy' : 'greedy-iter-1000 --deterministic-agent', 'safe' : 'safe-iter-100 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
    #           'corridor' : {'greedy' : 'greedy-iter-100 --deterministic-agent', 'safe' : 'safe-iter-4000 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
    #           'drone' : {'greedy' : 'greedy-iter-4000 --deterministic-agent', 'safe' : 'safe-iter-1000 --deterministic-agent', 'random' : 'random --uniform-random-policy'}} # USED FOR GENERATING COMMAND
    nus = {'dpm' : [0.2], 'corridor' : [0.2], 'drone' : [0.2], 'drone-b' : [0.2]} # USED FOR GENERATING COMMAND
    model_settings = {'dpm' : '--goal-rew 0.0 --fail-rew 0.0', 'corridor' : '', 'drone' : '', 'drone-b' : ''} # USED FOR GENERATING COMMAND
    shields = ['identity', 'standard', 'delta', 'pessimistic', 'optimistic', 'online', 'offline']

    # prepare what to run combinations
    eval_combinations = []
    model_list = models.keys()
    for model in model_list:
        agent_list = agents[model].keys()
        assert all(agent in agents[model].keys() for agent in agent_list), f"One or more specified eval agents are not valid for model {model}."
        for agent in agent_list:
            nu_list = nus[model]
            for nu in nu_list:
                for shield in shields:
                    eval_combinations.append( (model, agent, nu, shield) )

    for model, agent, nu, shield in tqdm.tqdm(eval_combinations):

        # shield init
        shield_string = f"--shield {shield} --shield-memory {shield_memory} --num-environments 2048 --num-parallel-environments 16 --min-episodes-per-environment 10"         # USED FOR GENERATING COMMAND
        shield_name = shield
        if shield in ['offline']:
            if shield_path == "":
                raise RuntimeError("Offline shield requires a shield path to be specified. (--shield-path)")
            shield_string = f"--load-shield {shield_path} --shield-memory {shield_memory} --num-environments 2048 --num-parallel-environments 16 --min-episodes-per-environment 10"
        if shield in ['online']:
            shield_string = f"--shield self-constructing-safe --shield-memory {shield_memory} --num-environments 2048 --num-parallel-environments 16 --min-episodes-per-environment 10"
        
        # Check if result already exists in df
        exists = (
            (df['model'] == model)
            & (df['agent'] == agent)
            & (df['nu'] == nu)
            & (df['shield'] == shield)
            & (df['shield_name'] == shield_name)
        ).any()

        if exists:
            print(f"Skipping existing result for model={model}, agent={agent}, nu={nu}, shield={shield}, shield_name={shield_name}")
            continue


        # Simulation-based evaluation
        total_risk = 0.0
        total_reward = 0.0
        total_shield_calls = 0
        total_blocked_actions = 0
        total_eval_time = 0.0
        wrong_results_detected = False
        for eval_iter in range(number_of_evaluations):
            command = f"python3 shielding.py {models[model]} --episode-length 50 --load-agent {agents[model][agent]} {shield_string} --nu {nu} {model_settings[model]} {log_settings}"

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
                wrong_results_detected = True
                break
            else:
                risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_time = [part.strip() for part in last_line.split(';', 5)]
                total_risk += float(risk)
                total_reward += float(reward)
                total_shield_calls += int(shield_calls)
                total_blocked_actions += int(blocked_actions)
                total_eval_time += float(eval_time)

        if wrong_results_detected:
            print(f"Skipping result for model={model}, agent={agent}, nu={nu}, shield={shield}, shield_name={shield_string} due to errors in evaluation.")
            continue

        avg_risk = total_risk / number_of_evaluations
        avg_reward = total_reward / number_of_evaluations
        avg_shield_calls = total_shield_calls / number_of_evaluations
        avg_blocked_actions = total_blocked_actions / number_of_evaluations
        avg_time_eval = total_eval_time / number_of_evaluations

        add_result_to_csv(main_csv_path, df, model, agent, nu, shield, shield_name, avg_risk, avg_reward, avg_shield_calls, avg_blocked_actions, earliest_shielded_step, avg_time_eval)



if __name__ == "__main__":
    main()
