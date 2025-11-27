import json
import matplotlib.pyplot as plt

import ast
import numpy as np

import os

def load_json_file(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def generate_plot(file1, file2, output_path):
    data1 = load_json_file(file1)
    data2 = load_json_file(file2)
    

    returns1 = data1["returns"]
    returns2 = data2["returns"]

    # Convert string of floats to floats
    returns1 = ast.literal_eval(returns1)
    returns2 = ast.literal_eval(returns2)
    
    steps = list(range(len(returns1)))
    plt.plot(steps, returns1, label="Masked Actions")
    plt.plot(steps, returns2, label="Unmasked Actions")
    plt.xlabel('i-th hundred training iteration')
    plt.ylabel('Average Return')
    plt.title('Network-3-8-20')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    file1 = 'models/models_pomdp_no_family/network-3-8-20/benchmark_stats_42.json'
    file2 = 'models/models_pomdp_no_family/network-3-8-20/benchmark_stats_42_0.json'
    output_path = 'refuel-10-reachability.png'
    generate_plot(file1, file2, output_path)
    
    

    