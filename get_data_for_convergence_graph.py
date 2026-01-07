import os
import click
import subprocess
import tqdm
import itertools


@click.command()
@click.option("--plot-data-file", type=str, required=True, help="Path to the plot data file.")
@click.option("--results-file", type=str, required=True, help="Path to the results file.")
@click.option("--shield-construction-type", type=click.Choice(['safe', 'unsafe']), default='safe', help="Shielding method to use.")
@click.option("--nu", type=float, required=True, help="Nu parameter for shielding.")
@click.option("--log-file", type=str, required=False, default=None, help="Path to the log file.")
@click.option("--model-path", type=str, required=True, help="Path to the model folder.")
@click.option("--episode-length", type=int, required=True, help="Maximum episode length.")
@click.option("--agent", type=str, required=True, help="Path to the saved agent.")
@click.option("--uniform-random-agent", is_flag=True, help="Use a uniform random agent instead of a trained agent.")
@click.option("--deterministic-agent", is_flag=True, default=False, help="Use a deterministic agent for evaluation.")
@click.option("--shield-file-name", type=str, required=False, default="", help="Name of the shield to use.")
@click.option("--shield-memory", type=int, required=False, default=0, help="Memory size for the shield.")
@click.option("--goal-rew", type=float, default=100.0, help="Reward value for reaching the goal state.")
@click.option("--fail-rew", type=float, default=0.0, help="Reward value for reaching the fail state.")
def main(plot_data_file, results_file, shield_construction_type, nu, log_file, model_path, episode_length, agent,
         uniform_random_agent, deterministic_agent, shield_file_name, shield_memory, goal_rew, fail_rew):

    # Ensure the directory for the results file exists
    results_dir = os.path.dirname(os.path.abspath(results_file))
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    if shield_construction_type == 'safe':
        shield_type = 'constructed-self-constructing-safe'
    elif shield_construction_type == 'unsafe':
        shield_type = 'constructed-self-constructing-unsafe'
    else:
        raise ValueError("Invalid shield construction type.")

    for current_iter in range(0, 256, 4):
        
        if shield_type in ["constructed-self-constructing-safe"]:
            command = f"python3 shielding.py {model_path} --episode-length {episode_length} --load-agent {agent} --shield self-constructing-safe --nu {nu} --model-checking-eval' {'--deterministic-agent' if deterministic_agent else ''} --goal-rew {goal_rew} --fail-rew {fail_rew} --shield-memory {shield_memory} --load-shield {agent+'-safe-'+(shield_file_name+'-' if shield_file_name != '' else '')+'mem_'+str(shield_memory)+'-'+str(nu).replace('.','')+'-'+('deterministic-' if deterministic_agent else '')+f'0-iter-{current_iter}-shield.pickle'} --eval-file {results_file}"
        elif shield_type in ["constructed-self-constructing-unsafe"]:
            command = f"python3 shielding.py {model_path} --episode-length {episode_length} --load-agent {agent} --shield self-constructing-unsafe --nu {nu} --model-checking-eval' {'--deterministic-agent' if deterministic_agent else ''} --goal-rew {goal_rew} --fail-rew {fail_rew} --shield-memory {shield_memory} --load-shield {agent+'-unsafe-'+(shield_file_name+'-' if shield_file_name != '' else '')+'mem_'+str(shield_memory)+'-'+str(nu).replace('.','')+'-'+('deterministic-' if deterministic_agent else '')+f'0-iter-{current_iter}-shield.pickle'} --eval-file {results_file}"

        if uniform_random_agent:
            command += " --uniform-random-policy"

            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            output, _ = process.communicate()
            if log_file:
                with open(log_file, "a") as f:
                    f.write(output.decode())

            # Process the last line of the output
            last_line = output.decode().strip().split('\n')[-1]
            if ';' not in last_line:
                print("Error: Output does not contain ';' in the last line.")
            else:
                safety, reward = [part.strip() for part in last_line.split(';', 1)]
                
            file_exists = os.path.exists(plot_data_file)
            with open(plot_data_file, "a") as f:
                if not file_exists:
                    f.write("model;agent;shield;shield_memory;nu;iter;safety;reward\n")
                actual_agent = "uniform-random" if uniform_random_agent else agent
                f.write(f"{os.path.basename(os.path.normpath(model_path))};{actual_agent};{shield_type};{shield_memory};{nu};{current_iter};{safety};{reward}\n")


if __name__ == "__main__":
    main()
