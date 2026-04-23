# Script to generate LaTeX table data from runtimes CSV
import csv
from collections import defaultdict, OrderedDict

CSV_PATH = 'test_data/31-03-runtimes.csv'

def main():
	# Read CSV and collect data
	models = set()
	shield_types = []
	data = defaultdict(dict)

	with open(CSV_PATH, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			model = row['model']
			shield = row['shield']
			models.add(model)
			if shield not in shield_types:
				shield_types.append(shield)
			try:
				shield_calls = float(row['shield_calls'])
				eval_time = float(row['eval_time'])
				memory_usage = float(row['memory_usage'])
				# value = shield_calls / eval_time if eval_time != 0 else 0.0 # shield calls per second
				# value = (eval_time / shield_calls) * 1000000 if shield_calls != 0 else 0.0 # runtime normalized to 1M shield calls
				value = memory_usage
			except Exception:
				value = '-'
			data[model][shield] = value

	desired_order = ['corridor', 'dpm', 'drone', 'drone-b']
	models = [m for m in desired_order if m in models]
	# Remove 'identity' shield from columns
	# shield_types = [s for s in shield_types if s and s != 'identity']

	print("Model & " + " & ".join(shield_types) + " \\\\")
	print("\\midrule")

	# Print LaTeX table using booktabs
	for model in models:
		row = [model]
		for shield in shield_types:
			val = data[model].get(shield, '-')
			if isinstance(val, float):
				val = f"{val:.1f}"
			row.append(str(val))
		print(' & '.join(row) + ' \\\\')

if __name__ == '__main__':
	main()
