import os
import json
import re

import ast

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_all_benchmarks_from_folder(folder_path):
    benchmark_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            if "evaluation" not in filename:
                continue
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                benchmark_data.append(data)
    return benchmark_data

def process_benchmark_data(benchmark_data):
    reachabilities_by_method = {}
    returns_by_method = {}
    for data in benchmark_data:
        for eval_option, reachabilities in data["reachabilities_per_evaluation_option"].items():
            if eval_option not in reachabilities_by_method:
                reachabilities_by_method[eval_option] = []
            reachabilities_by_method[eval_option].append(ast.literal_eval(reachabilities))
        for eval_option, returns in data["returns_per_evaluation_option"].items():
            if eval_option not in returns_by_method:
                returns_by_method[eval_option] = []
            returns_by_method[eval_option].append(ast.literal_eval(returns))
    return {"reachabilities": reachabilities_by_method, "returns": returns_by_method}

def plot_single_convergence_curve(data, model_name, output_path, metric_name):
    plt.figure(figsize=(10, 6))
    for eval_option, runs in data.items():
        mean_values = pd.DataFrame(runs).mean()
        upper_values = pd.DataFrame(runs).quantile(0.95)
        lower_values = pd.DataFrame(runs).quantile(0.05)
        plt.fill_between(range(len(mean_values)), lower_values, upper_values, alpha=0.2)
        plt.plot(mean_values, label=eval_option)
    
    plt.title(f"{metric_name} Convergence Curve for {model_name}")
    plt.xlabel("Training Iterations")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt_path = os.path.join(output_path, f"{model_name}_{metric_name.lower()}_convergence_curve.png")
    plt.savefig(plt_path)
    plt.close()

def plot_evaluation_curves(processed_data, model_name, output_path):
    # Processed data contains 'reachabilities' and 'returns' dictionaries, each mapping evaluation options to lists of independent seeded runs.
    print(f"Generating convergence curves for model: {model_name}")
    pdf = pd.DataFrame(processed_data)
    num_training_iterations = pdf["reachabilities"]["FULL_STOCHASTIC"][0].__len__()
    
    plot_single_convergence_curve(pdf["reachabilities"], model_name, output_path, "Reachability")
    plot_single_convergence_curve(pdf["returns"], model_name, output_path, "Return")
    


def generate_evaluation_type_curves(models_folder, output_path):
    for model_name in os.listdir(models_folder):
        model_folder = os.path.join(models_folder, model_name)
        model_data = load_all_benchmarks_from_folder(model_folder)
        processed_data = process_benchmark_data(model_data)
        plot_evaluation_curves(processed_data, model_name, output_path)
    
if __name__ == "__main__":
    models_folder = "models/models_robust/"
    output_path = "plots/evaluation_type_curves/"
    os.makedirs(output_path, exist_ok=True)
    generate_evaluation_type_curves(models_folder, output_path)