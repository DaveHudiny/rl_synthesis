import re
import os

import json
import ast

import numpy as np

import pandas as pd

def load_all_benchmark_stats_from_folder(directory):
    # Load all benchmark stats from JSON files in the current directory
    
    # Loop through all files in the directory
    benchmark_stats = []
    for filename in os.listdir(directory):
        sequences_of_numbers = re.findall(r'\d+', filename)
        at_least_one_four_digit = any(len(num) >= 4 for num in sequences_of_numbers)
        if not at_least_one_four_digit:
            continue
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    benchmark_stats.append((filename, data))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return benchmark_stats

def load_all_benchmark_directories(base_directory):
    # Load all benchmark stats from JSON files in the current directory
    benchmark_stats = {}
    for entry in os.listdir(base_directory):
        full_path = os.path.join(base_directory, entry)
        if os.path.isdir(full_path):
            stats = load_all_benchmark_stats_from_folder(full_path)
            benchmark_stats[entry] = stats
    return benchmark_stats

def get_rfgp_values_constants():
    return {
        "obstacles-8-5": 205.86,
        "rover": 296.85,
        "network": 3.48,
        "maze-10": 7.04,
        "drone": float('NaN'),
        "avoid": 161.05,
    }

MAX_MODELS = ["network", "drone-2-6-1", "maze-10", "rover", "drone-2-6-1-large-fam"]
            
def generate_latex_table_from_stats(benchmark_stats):
    "We are interested in family_performance"
    model_family_performance = {}
    model_fsc_sizes = {}
    for model in benchmark_stats:
        stats = benchmark_stats[model]
        records = {}
        records_fsc_size = {}
        for entry in stats:
            if "family_performance" in entry[1] and entry[1]["extraction_less"]==False:
                if entry[1]["lstm_extracted_reachability"] != "[]":
                    continue

                # Find maximum performance in the list
                list_converted = ast.literal_eval(entry[1]["family_performance"])
                operator = max if model in MAX_MODELS else min
                max_performance = operator(list_converted[i] for i in range(len(list_converted)))
                arg_index_max_performance = list_converted.index(max_performance)

                if "extraction_type" in entry[1]:
                    extraction_type = entry[1]["extraction_type"]
                else:
                    extraction_type = "unknown"
                if extraction_type not in records:
                    records[extraction_type] = []
                records[extraction_type].append(max_performance)
                fsc_nodes = eval(entry[1]["available_nodes_in_fsc"].replace("array", "np.array"))
                fsc_sizes = [np.sum(fsc_nodes[i]) for i in range(len(fsc_nodes))]
                if extraction_type not in records_fsc_size:
                    records_fsc_size[extraction_type] = []
                records_fsc_size[extraction_type].append(fsc_sizes[arg_index_max_performance])

        model_family_performance[model] = records
        model_fsc_sizes[model] = records_fsc_size
    # Now create a LaTeX table
    pre_df = {"Model": []}
    
    extraction_types = set()
    for model in model_family_performance:
        extraction_types.update(model_family_performance[model].keys())
    extraction_types = sorted(extraction_types)
    for ext_type in extraction_types:
        pre_df[f"M ({ext_type})"] = []
        pre_df[f"S ({ext_type})"] = []
        pre_df[f"B ({ext_type})"] = []
        pre_df[f"Average FSC Size ({ext_type})"] = []
    pre_df[f"Goal"] = []
    for model in model_family_performance:
        is_max = model in MAX_MODELS
        pre_df[f"Goal"].append("Max" if is_max else "Min")
    for model in model_family_performance:
        operator = max if model in MAX_MODELS else min
        pre_df["Model"].append(model)
        for ext_type in extraction_types:
            if ext_type in model_family_performance[model]:
                values = model_family_performance[model][ext_type]
                mean_val = sum(values) / len(values)
                std_dev = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                pre_df[f"M ({ext_type})"].append(f"{mean_val:.2f}")
                pre_df[f"S ({ext_type})"].append(f"{std_dev:.2f}")
                pre_df[f"B ({ext_type})"].append(f"{operator(values):.2f}")
                fsc_sizes = model_fsc_sizes[model][ext_type]
                mean_fsc_size = sum(fsc_sizes) / len(fsc_sizes)
                pre_df[f"Average FSC Size ({ext_type})"].append(f"{mean_fsc_size:.2f}")


            else:
                pre_df[f"M ({ext_type})"].append("N/A")
                pre_df[f"S ({ext_type})"].append("N/A")
                pre_df[f"B ({ext_type})"].append("N/A")
                pre_df[f"Average FSC Size ({ext_type})"].append("N/A")


    df = pd.DataFrame(pre_df)
    print(df.style.to_latex(column_format='|' + 'c|' * len(df.columns), hrules=True, sparse_index=True))
            


if __name__ == "__main__":
    base_directory = "models/models_table_camera_ready"
    benchmark_stats = load_all_benchmark_directories(base_directory)
    # print(benchmark_stats["network"][0][1]) # Access of one benchmark stats from the "network" directory
    generate_latex_table_from_stats(benchmark_stats)
    
    base_directory = "models_single_pomdp"
    benchmark_stats = load_all_benchmark_directories(base_directory)
    # print(benchmark_stats["network"][0][1]) # Access of one benchmark stats from the "network" directory
    generate_latex_table_from_stats(benchmark_stats)
