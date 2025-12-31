import os
import click
import subprocess
import tqdm
import itertools

# Example of use:
# python3 run_benchmark.py --results-file results/test-corridor-greedy-test.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 4096 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.1 --number-of-evaluations 3

@click.command()
@click.option("--results-file", type=str, required=True, help="Path to the results file.")
@click.option("--log-file", type=str, required=False, default=None, help="Path to the log file.")
@click.option("--model-path", type=str, required=True, help="Path to the model folder.")
@click.option("--episode-length", type=int, required=True, help="Maximum episode length.")
@click.option("--num-environments", type=int, required=True, help="Number of total environments.")
@click.option("--num-parallel-environments", type=int, required=True, help="Number of parallel environments.")
@click.option("--agent", type=str, required=True, help="Path to the saved agent.")
@click.option("--nu", type=float, required=True, help="Nu parameter for shielding.")
@click.option("--number-of-evaluations", type=int, required=True, help="Number of evaluations to run for each shield.")
@click.option("--uniform-random-agent", is_flag=True, help="Use a uniform random agent instead of a trained agent.")
@click.option("--model-checking-eval", is_flag=True, default=False, help="Whether to perform model checking based evaluation.")
@click.option("--goal-rew", type=float, default=100.0, help="Reward value for reaching the goal state.")
@click.option("--fail-rew", type=float, default=0.0, help="Reward value for reaching the fail state.")
@click.option("--deterministic-agent", is_flag=True, default=False, help="Use a deterministic agent for evaluation.")
def main(results_file, log_file, model_path, episode_length, num_environments,
         num_parallel_environments, agent, nu, number_of_evaluations, uniform_random_agent, model_checking_eval, goal_rew, fail_rew, deterministic_agent):

    # Ensure the directory for the results file exists
    results_dir = os.path.dirname(os.path.abspath(results_file))
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    considerred_shields = ["identity", "standard", "delta", "pessimistic", "optimistic", "self-constructing-safe", "self-constructing-unsafe", "constructed-self-constructing-safe", "constructed-self-constructing-unsafe"]

    shield_iteration_combinations = list(itertools.product(considerred_shields, range(number_of_evaluations)))

    for shield_type, i in tqdm.tqdm(shield_iteration_combinations):
        
        if shield_type in ["identity", "standard", "delta", "pessimistic", "optimistic"]:
            command = f"python3 shielding.py {model_path} --episode-length {episode_length} --num-environments {num_environments} --num-parallel-environments {num_parallel_environments} --load-agent {agent} --shield {shield_type} --nu {nu} {'--model-checking-eval' if model_checking_eval else ''} {'--deterministic-agent' if deterministic_agent else ''} --goal-rew {goal_rew} --fail-rew {fail_rew} --eval-file {results_file}"
        elif shield_type in ["self-constructing-safe"]:
            command = f"python3 shielding.py {model_path} --episode-length {episode_length} --num-environments {num_environments} --num-parallel-environments {max(num_parallel_environments,16)} --load-agent {agent} --shield {shield_type} --nu {nu} {'--model-checking-eval' if model_checking_eval else ''} {'--deterministic-agent' if deterministic_agent else ''} --goal-rew {goal_rew} --fail-rew {fail_rew} --save-shield {agent+'-safe-'+str(nu).replace('.','')+'-'+str(i)} --eval-file {results_file}"
        elif shield_type in ["self-constructing-unsafe"]:
            command = f"python3 shielding.py {model_path} --episode-length {episode_length} --num-environments {num_environments} --num-parallel-environments {num_parallel_environments} --load-agent {agent} --shield {shield_type} --nu {nu} {'--model-checking-eval' if model_checking_eval else ''} {'--deterministic-agent' if deterministic_agent else ''} --goal-rew {goal_rew} --fail-rew {fail_rew} --save-shield {agent+'-unsafe-'+str(nu).replace('.','')+'-'+str(i)} --eval-file {results_file}"
        elif shield_type in ["constructed-self-constructing-safe"]:
            command = f"python3 shielding.py {model_path} --episode-length {episode_length} --num-environments {num_environments} --num-parallel-environments {num_parallel_environments} --load-agent {agent} --shield self-constructing-safe --nu {nu} {'--model-checking-eval' if model_checking_eval else ''} {'--deterministic-agent' if deterministic_agent else ''} --goal-rew {goal_rew} --fail-rew {fail_rew} --load-shield {agent+'-safe-'+str(nu).replace('.','')+'-'+'0-iter-final-shield.pickle'} --eval-file {results_file}"
        elif shield_type in ["constructed-self-constructing-unsafe"]:
            command = f"python3 shielding.py {model_path} --episode-length {episode_length} --num-environments {num_environments} --num-parallel-environments {num_parallel_environments} --load-agent {agent} --shield self-constructing-unsafe --nu {nu} {'--model-checking-eval' if model_checking_eval else ''} {'--deterministic-agent' if deterministic_agent else ''} --goal-rew {goal_rew} --fail-rew {fail_rew} --load-shield {agent+'-unsafe-'+str(nu).replace('.','')+'-'+'0-iter-final-shield.pickle'} --eval-file {results_file}"

        if uniform_random_agent:
            command += " --uniform-random-policy"

        if log_file:
            with open(log_file, "a") as f:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                output, _ = process.communicate()
                f.write(output.decode())
        else:
            subprocess.run(command, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        


if __name__ == "__main__":
    main()
