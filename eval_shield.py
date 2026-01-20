import os
import click
import subprocess
import tqdm
import itertools
import pandas as pd


def add_result_to_csv(csv_path, df, model, agent, nu, shield, shield_name, risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_time):
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
@click.option("--shield", type=click.Choice(['identity', 'standard', 'delta', 'pessimistic', 'optimistic', 'online', 'offline']), required=True, help="Type of shield to evaluate.")
@click.option("--shield-memory", type=int, required=False, default=0, help="Memory size for the shield.")
@click.option("--shield-path", type=str, required=False, default="", help="Path to the shield file if applicable.")
@click.option("--number-of-evaluations", type=int, required=False, default=3, help="Number of evaluations to run for the shield.")
@click.option("--shield-nu", type=float, required=False, default=None, help="Nu parameter for shielding.")
@click.option("--shield-model", type=str, required=False, default=None, help="Model to use for evaluation: 'dpm', 'corridor', or 'drone'.")
@click.option("--shield-construction-agent", type=str, required=False, default=None, help="Agent used for shield construction if applicable.")
@click.option("--eval-agent", type=str, required=False, default=None, help="Agent used for evaluation.")
def main(results_folder, shield, shield_memory, shield_path, number_of_evaluations, shield_nu, shield_model, shield_construction_agent, eval_agent):

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

    script_raw_results_path = os.path.join(results_folder, "raw_evaluation_results.csv")
    script_log_path = os.path.join(results_folder, "evaluation_log.log")
    log_settings = f"--eval-file {script_raw_results_path}" # USED FOR GENERATING COMMAND

    # init eval parameters
    models = {'dpm' : 'models/shielding/dpm', 'corridor' : 'models/shielding/test-corridor', 'drone' : 'models/shielding/slippy-drone'} # USED FOR GENERATING COMMAND
    agents = {'dpm' : {'greedy' : 'greedy-iter-1000 --deterministic-agent', 'safe' : 'safe-iter-100 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'corridor' : {'greedy' : 'greedy-iter-100 --deterministic-agent', 'safe' : 'safe-iter-4000 --deterministic-agent', 'random' : 'random --uniform-random-policy'},
              'drone' : {'greedy' : 'greedy-iter-4000 --deterministic-agent', 'safe' : 'safe-iter-1000 --deterministic-agent', 'random' : 'random --uniform-random-policy'}} # USED FOR GENERATING COMMAND
    nus = {'dpm' : [0.01, 0.05, 0.2], 'corridor' : [0.05, 0.1, 0.2], 'drone' : [0.01, 0.05, 0.2]} # USED FOR GENERATING COMMAND
    model_settings = {'dpm' : '--goal-rew 0.0 --fail-rew 0.0', 'corridor' : '', 'drone' : ''} # USED FOR GENERATING COMMAND

    # shield init
    depends_on_nu = False
    custom_nu = False
    custom_model = False
    simulation_eval = False
    shield_string = f"--shield {shield} --shield-memory {shield_memory}"         # USED FOR GENERATING COMMAND
    shield_name = shield
    if shield in ['delta', 'pessimistic', 'optimistic', 'online', 'offline']:
        depends_on_nu = True
    if shield in ['offline']:
        if shield_path == "":
            raise RuntimeError("Offline shield requires a shield path to be specified. (--shield-path)")
        shield_string = f"--load-shield {shield_path} --shield-memory {shield_memory}"
        assert shield_nu is not None, "Offline shield requires a nu parameter to be specified. (--shield-nu)"
        custom_nu = True
        assert shield_model in ['dpm', 'corridor', 'drone'], "Offline shield requires a model to be specified. (--shield-model)"
        custom_model = True
        assert shield_construction_agent is not None, "Offline shield requires a shield construction agent to be specified. (--shield-construction-agent)"
        shield_name = f"{shield_construction_agent}"
    if shield in ['online']:
        shield_string = f"--shield self-constructing-safe --shield-memory {shield_memory} --num-environments 2048 --num-parallel-environments 16 --min-episodes-per-environment 10"
    if shield in ['pessimistic', 'optimistic', 'online']:
        simulation_eval = True
    if shield in ['pessimistic', 'optimistic']:
        shield_string = f"--shield {shield} --shield-memory {shield_memory}  --num-environments 2048 --num-parallel-environments 512 --min-episodes-per-environment 10"

    # prepare what to run combinations
    eval_combinations = []
    model_list = models.keys() if not custom_model else [shield_model]
    for model in model_list:
        agent_list = agents[model].keys() if eval_agent is None else [eval_agent]
        assert all(agent in agents[model].keys() for agent in agent_list), f"One or more specified eval agents are not valid for model {model}."
        for agent in agent_list:
            nu_list = nus[model] if depends_on_nu and not custom_nu else ([shield_nu] if custom_nu else [nus[model][0]])
            for nu in nu_list:
                eval_combinations.append( (model, agent, nu) )

    for model, agent, nu in tqdm.tqdm(eval_combinations):
        
        # Check if result already exists in df
        exists = (
            (df['model'] == model)
            & (df['agent'] == agent)
            & (df['nu'] == nu)
            & (df['shield'] == shield)
            & (df['shield_name'] == shield_name)
        ).any()

        if not depends_on_nu:
            # Also check for other nus
            for other_nu in nus[model]:
                if other_nu != nu:
                    exists &= (
                        (df['model'] == model)
                        & (df['agent'] == agent)
                        & (df['nu'] == other_nu)
                        & (df['shield'] == shield)
                        & (df['shield_name'] == shield_name)
                    ).any()

        if exists:
            print(f"Skipping existing result for model={model}, agent={agent}, nu={nu}, shield={shield}, shield_name={shield_name}")
            if not depends_on_nu:
                # Also skip for other nus
                for other_nu in nus[model]:
                    if other_nu != nu:
                        print(f"Skipping existing result for model={model}, agent={agent}, nu={other_nu}, shield={shield}, shield_name={shield_name}")
            continue

        if not simulation_eval:
            command = f"python3 shielding.py {models[model]} --episode-length 50 --load-agent {agents[model][agent]} {shield_string} --nu {nu} {model_settings[model]} {log_settings} --model-checking-eval --expected-shield-calls"

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
                risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_elapsed_time = [part.strip() for part in last_line.split(';', 5)]

                add_result_to_csv(main_csv_path, df, model, agent, nu, shield, shield_name, risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_elapsed_time)
                
                if not depends_on_nu:
                    # Also add for other nus
                    for other_nu in nus[model]:
                        if other_nu != nu:
                            add_result_to_csv(main_csv_path, df, model, agent, other_nu, shield, shield_name, risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_elapsed_time)
        else:
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
                    risk, reward, shield_calls, blocked_actions, earliest_shielded_step, eval_elapsed_time = [part.strip() for part in last_line.split(';', 4)]
                    total_risk += float(risk)
                    total_reward += float(reward)
                    total_shield_calls += int(shield_calls)
                    total_blocked_actions += int(blocked_actions)
                    total_eval_time += float(eval_elapsed_time)

            if wrong_results_detected:
                print(f"Skipping result for model={model}, agent={agent}, nu={nu}, shield={shield}, shield_name={shield_string} due to errors in evaluation.")
                continue

            avg_risk = total_risk / number_of_evaluations
            avg_reward = total_reward / number_of_evaluations
            avg_shield_calls = total_shield_calls / number_of_evaluations
            avg_blocked_actions = total_blocked_actions / number_of_evaluations

            add_result_to_csv(main_csv_path, df, model, agent, nu, shield, shield_name, avg_risk, avg_reward, avg_shield_calls, avg_blocked_actions, earliest_shielded_step, total_eval_time)



if __name__ == "__main__":
    main()
