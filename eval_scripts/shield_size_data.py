

from rl_src.shielding.constructed_shield_data import ShieldData


shield_paths = {('dpm', 'greedy'): '../trained_agents/shields/dpm/greedy-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('dpm', 'timid'): '../trained_agents/shields/dpm/safe-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('dpm', 'random'): '../trained_agents/shields/dpm/random-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('corridor', 'greedy'): '../trained_agents/shields/test-corridor/greedy-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('corridor', 'timid'): '../trained_agents/shields/test-corridor/safe-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('corridor', 'random'): '../trained_agents/shields/test-corridor/random-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('drone', 'greedy'): '../trained_agents/shields/slippy-drone/greedy-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('drone', 'timid'): '../trained_agents/shields/slippy-drone/safe-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('drone', 'random'): '../trained_agents/shields/slippy-drone/random-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('drone-b', 'greedy'): '../trained_agents/shields/collect/greedy-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('drone-b', 'timid'): '../trained_agents/shields/collect/safe-mem_0-nu_02--eval_0-iter-final-shield.pickle', ('drone-b', 'random'): '../trained_agents/shields/collect/random-mem_0-nu_02--eval_0-iter-final-shield.pickle'}


import os
import pickle

def get_file_size_mb(path):
	try:
		return os.path.getsize(path) / (1024 * 1024)
	except Exception:
		return None

def get_node_count(path):
	try:
		with open(path, 'rb') as f:
			x = pickle.load(f)
			return x.initial_node.number_of_tree_nodes()
	except Exception:
		return None

def main():
	models = ['dpm', 'corridor', 'drone', 'drone-b']
	agents = ['greedy', 'timid', 'random']

	# Gather data
	size_data = {model: {} for model in models}
	node_data = {model: {} for model in models}
	for model in models:
		for agent in agents:
			path = shield_paths.get((model, agent))
			if path:
				size = get_file_size_mb(path)
				nodes = get_node_count(path)
				size_data[model][agent] = size
				node_data[model][agent] = nodes
			else:
				size_data[model][agent] = None
				node_data[model][agent] = None

	# Print LaTeX tabular for file sizes
	print('% File sizes in MB')
	print('Model      & greedy   & timid    & random   \\')
	for model in models:
		row = [model]
		for agent in agents:
			val = size_data[model][agent]
			row.append(f'{val:.2f}' if val is not None else '-')
		print(' & '.join(row) + ' \\')

	print('\n% Number of nodes')
	print('Model      & greedy   & timid    & random   \\')
	for model in models:
		row = [model]
		for agent in agents:
			val = node_data[model][agent]
			row.append(str(val) if val is not None else '-')
		print(' & '.join(row) + ' \\')

if __name__ == "__main__":
	main()


