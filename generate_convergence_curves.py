import os
import json
import re

import ast

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# If you have your own experiments, replace the constants with your own values.
CONSTANTS_SINGLE_PAYNT_SIG = {
    "drone-2-8-1": 0.58,
    "intercept-16": 0.99,
    "evade-n17": 0.85,
    "rocks-16": 48.56,
    "network-3-8-20": 7.66,
    "network-5-10-8": 12.71,
    "maze-10": 8.48,
}

CONSTANTS_SINGLE_PAYNT_ALERGIA = {
    "drone-2-8-1": 0.58,
    "intercept-16": 0.98,
    "evade-n17": 0.85,
    "rocks-16": 51.72,
    "network-3-8-20": 8.05,
    "network-5-10-8": 13.86,
    "maze-10": 8.51,
}


def load_all_benchmark_stats_from_folder(directory, training_include=False):
    # Load all benchmark stats from JSON files in the current directory

    # Loop through all files in the directory
    benchmark_stats = []
    for filename in os.listdir(directory):
        sequences_of_numbers = re.findall(r'\d+', filename)
        at_least_one_four_digit = any(
            len(num) >= 4 for num in sequences_of_numbers)
        if "args" in filename:
            continue
        if "training" in filename and not training_include:
            continue
        if "training" not in filename and training_include:
            continue
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


def load_all_benchmark_directories(base_directory, training_include=False):
    # Load all benchmark stats from JSON files in the current directory
    benchmark_stats = {}
    for entry in os.listdir(base_directory):
        full_path = os.path.join(base_directory, entry)
        if os.path.isdir(full_path):
            stats = load_all_benchmark_stats_from_folder(
                full_path, training_include=training_include)
            benchmark_stats[entry] = stats
    return benchmark_stats


def use_reward(model_name):
    return model_name not in ["drone-2-6-1", "drone-2-6-1-large-fam"]


def use_reward_single(model_name):
    return model_name in ["rocks-16", "network-3-8-20", "network-5-10-8", "maze-10"]


def is_negative(model_name):
    return model_name in ["obstacles-8-5", "avoid", "moving_obstacles-8-5", "moving-obstacles-xl", "avoid-large", "moving-obstacles-6-3-xl", "moving-obstacles-large-fam", "moving-obstacles"]


def is_negative_single(model_name):
    return model_name in ["rocks-16", "network-3-8-20", "network-5-10-8"]


def get_iterations_rfpg(model_name):
    print("Getting rfpg iterations for model:", model_name)
    rfpg_iterations = {
        "moving-obstacles-large-fam": [[2290.0248720467193, 2100.2244708653598], [2505.378902937115, 2400.150555216214],
                                       [2348.992497412948, 2107.5366790865933],
                                       [2131.9061349791937], [2898.884866206211, 2376.5245425056637]]
    }
    return rfpg_iterations.get(model_name, [])


def update_records(model, records, entry, gru_extracted=False):
    extraction_less = entry[1]["extraction_less"] if "extraction_less" in entry[1] else False
    if entry[1]["extraction_type"] == "bottleneck":
        return
    time_limit = 3600 if not gru_extracted else 10000
    # check family_performance_times and find the last time that is <= time_limit
    times = ast.literal_eval(entry[1]["family_performance_times"])
    last_index = 0
    for i in range(len(times)):
        if times[i] <= time_limit:
            last_index = i
        else:
            break
    if use_reward(model):
        rl_values = ast.literal_eval(
            entry[1]["average_rl_return_subset_simulated"])[:last_index+1]
        extracted_values = ast.literal_eval(
            entry[1]["average_extracted_fsc_return_subset_simulated"])[:last_index+1]
        gru_extracted_values = ast.literal_eval(
            entry[1]["lstm_extracted_return"])[:last_index+1] if "lstm_extracted_return" in entry[1] else []
        for i in range(len(extracted_values)):
            rl_values[i] = abs(rl_values[i])
            extracted_values[i] = abs(extracted_values[i])
            if i < len(gru_extracted_values):
                gru_extracted_values[i] = abs(gru_extracted_values[i])
                records["GRU Extracted"].append(gru_extracted_values)
    else:
        rl_values = ast.literal_eval(
            entry[1]["average_rl_reachability_subset_simulated"])[:last_index+1]
        extracted_values = ast.literal_eval(
            entry[1]["average_extracted_fsc_reachability_subset_simulated"])[:last_index+1]

    verified_values = ast.literal_eval(
        entry[1]["family_performance"])[:last_index+1]
    if not extraction_less:
        records[f"RL {entry[1]['extraction_type']}"].append(rl_values)
        records[f"Extracted FSC {entry[1]['extraction_type']}"].append(
            extracted_values)
        records[f"Synthesized Worst-Case {entry[1]['extraction_type']}"].append(
            verified_values)
    else:
        records[f"RL extraction_less {entry[1]['extraction_type']}"].append(
            rl_values)
        records[f"Extracted FSC extraction_less {entry[1]['extraction_type']}"].append(
            extracted_values)
        records[f"Synthesized Worst-Case extraction_less {entry[1]['extraction_type']}"].append(
            verified_values)


def generate_convergence_curves(benchmark_stats, output_directory, gru_extracted=False):
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
    for model in benchmark_stats:
        stats = benchmark_stats[model]
        records = {
            "Synthesized Worst-Case si-g": [],
            "Synthesized Worst-Case alergia": [],
            "Synthesized Worst-Case extraction_less alergia": [],
            "Synthesized Worst-Case extraction_less si-g": [],
            "RL si-g": [],
            "RL alergia": [],
            "RL extraction_less alergia": [],
            "RL extraction_less si-g": [],
            "Extracted FSC si-g": [],
            "Extracted FSC alergia": [],
            "Extracted FSC extraction_less alergia": [],
            "Extracted FSC extraction_less si-g": [],
            "Extraction Method": [],
            "GRU Extracted": [],
        }

        for entry in stats:
            # if entry[1]["extraction_type"] == "si-g":
            # continue
            update_records(model, records, entry, gru_extracted=gru_extracted)

            # records["GRU Extracted"].append(gru_extracted_values)
        if is_negative(model):
            for key in records:
                for i in range(len(records[key])):
                    records[key][i] = [-v for v in records[key][i]]
        max_length = max(len(v) for values in records.values() for v in values)
        data = []
        for label, values_list in records.items():
            if "FSC Size" in label:
                continue
            method = "si-g" if "si-g" in label else (
                "alergia" if "alergia" in label else "extraction_less")
            for i, values in enumerate(values_list):
                padded = values + [values[-1]] * (max_length - len(values))
                for iteration, value in enumerate(padded):
                    data.append({
                        "Iteration": iteration,
                        "Value": value,
                        "Method": label
                    })
        df = pd.DataFrame(data)
        if model == "drone-2-6-1":
            df_copy = df.copy()
            plot_robust_lineplot_drone(
                df_copy, f"{model}_main", output_directory)
        if model == "moving-obstacles":
            df_copy = df.copy()
            plot_robust_moving(df_copy, f"{model}_main", output_directory)
        if gru_extracted:
            df_copy = df.copy()
            plot_robust_lineplot(df_copy, f"{model}_gru", output_directory)
        else:
            plot_robust_lineplot(df, model, output_directory)


def renamer(key):
    rename_dict = {
        "Synthesized Worst-Case si-g": "Worst-Case SIG",
        "Synthesized Worst-Case alergia": "Worst-Case Alergia",
        "Synthesized Worst-Case extraction_less alergia": "Worst-Case Random",
        "Synthesized Worst-Case extraction_less si-g": "Worst-Case Random",
        "Empirical RL Performance": "Empirical RL Values",
        "RL si-g": "Empirical RL SIG",
        "RL alergia": "Empirical RL Alergia",
        "RL extraction_less alergia": "Empirical RL Random",
        "RL extraction_less si-g": "Empirical RL Random",
        "Extracted FSC si-g": "Extracted FSC SIG",
        "Extracted FSC alergia": "Extracted FSC Alergia",
    }
    return rename_dict.get(key, key)


def remove_useless_data(df):
    df = df[df["Method"] != "rfPG"]
    # df = df[df["Method"] != "Worst-Case SIG Naive"]
    # df = df[df["Method"] != "Worst-Case Alergia Naive"]
    # df = df[df["Method"] != "Extracted FSC si-g"]
    # df = df[df["Method"] != "Extracted FSC alergia"]
    df = df[df["Method"] != "Extracted FSC extraction_less alergia"]
    df = df[df["Method"] != "Extracted FSC extraction_less si-g"]
    # df = df[df["Method"] != "Worst-Case Robust Lexipop SIG"]
    df = df[df["Method"] != "Worst-Case Random"]
    # df = df[df["Method"] != "Empirical RL Lexipop si-g"]
    df = df[df["Method"] != "Empirical RL Random"]
    # df = df[df["Method"] != "Empirical RL SIG"]
    # df = df[df["Method"] != "Worst-Case SIG"]
    # df = df[df["Method"] != "Empirical RL Alergia"]
    # df = df[df["Method"] != "Worst-Case Alergia"]
    df = df[df["Method"] != "FSC Size si-g"]
    df = df[df["Method"] != "FSC Size alergia"]
    return df


def remove_data_general(df):
    df = df[df["Method"] != "rfPG"]
    df = df[df["Method"] != "Extracted FSC extraction_less alergia"]
    df = df[df["Method"] != "Extracted FSC extraction_less si-g"]
    df = df[df["Method"] != "Extracted FSC si-g"]
    df = df[df["Method"] != "Extracted FSC alergia"]
    df = df[df["Method"] != "Worst-Case Random"]
    df = df[df["Method"] != "Empirical RL Random"]
    df = df[df["Method"] != "FSC Size si-g"]
    df = df[df["Method"] != "FSC Size alergia"]
    return df


def remove_data_obstacles(df):
    df = df[df["Method"] != "rfPG"]
    # df = df[df["Method"] != "Worst-Case SIG Naive"]
    # df = df[df["Method"] != "Worst-Case Alergia Naive"]
    df = df[df["Method"] != "Extracted FSC si-g"]
    # df = df[df["Method"] != "Extracted FSC alergia"]
    df = df[df["Method"] != "Extracted FSC extraction_less alergia"]
    df = df[df["Method"] != "Extracted FSC extraction_less si-g"]
    # df = df[df["Method"] != "Worst-Case Robust Lexipop SIG"]
    df = df[df["Method"] != "Worst-Case Random"]
    # df = df[df["Method"] != "Empirical RL Lexipop si-g"]
    df = df[df["Method"] != "Empirical RL Random"]
    # df = df[df["Method"] != "Empirical RL SIG"]
    # df = df[df["Method"] != "Worst-Case SIG"]
    # df = df[df["Method"] != "Empirical RL Alergia"]
    # df = df[df["Method"] != "Worst-Case Alergia"]
    df = df[df["Method"] != "FSC Size si-g"]
    df = df[df["Method"] != "FSC Size alergia"]
    df = df[df["Method"] != "Extracted FSC Alergia"]
    df = df[df["Method"] != "Extracted FSC SIG"]
    return df


def remove_data_drone(df):
    df = df[df["Method"] != "rfPG"]
    df = df[df["Method"] != "Extracted FSC extraction_less alergia"]
    df = df[df["Method"] != "Extracted FSC extraction_less si-g"]
    df = df[df["Method"] != "FSC Size si-g"]
    df = df[df["Method"] != "FSC Size alergia"]
    df = df[df["Method"] != "Worst-Case Alergia"]
    df = df[df["Method"] != "Empirical RL Alergia"]
    df = df[df["Method"] != "Extracted FSC Alergia"]
    df = df[df["Method"] != "Extracted FSC SIG"]
    return df


def remove_data_gru_experiment(df):
    df = df[df["Method"] != "rfPG"]
    df = df[df["Method"] != "Extracted FSC extraction_less alergia"]
    df = df[df["Method"] != "Extracted FSC extraction_less si-g"]
    # df = df[df["Method"] != "Extracted FSC si-g"]
    df = df[df["Method"] != "Extracted FSC alergia"]
    df = df[df["Method"] != "Worst-Case Random"]
    df = df[df["Method"] != "Empirical RL Random"]
    df = df[df["Method"] != "FSC Size si-g"]
    df = df[df["Method"] != "FSC Size alergia"]
    return df


def plot_robust_lineplot(df, model, output_directory):
    df["Method"] = df["Method"].apply(renamer)
    if "gru" in model:
        df = remove_data_gru_experiment(df)
    else:
        df = remove_data_general(df)
    palette = sns.color_palette("tab10")
    # Palette = green, blue, red

    if "gru" in model:
        linestyles = [
            (),                  # "-" solid
            (5, 5),              # "--"
            (5, 5),              # "--"
            (5, 5),              # "--"
        ]
        palette = ["green", "green", "orange", "black"]
    else:
        linestyles = [
            (),                  # "-" solid
            (),                  # "-" solid
            (5, 5),              # "--"
            (5, 5),              # "--"
            (1, 1),              # ":" dotted
            (1, 1),              # ":" dotted
        ]
        palette = ["green", "red", "green", "red", "green", "red"]

    dashes_map = {m: ls for m, ls in zip(df["Method"].unique(), linestyles)}

    plt.figure(figsize=(3.9, 3.9))
    ax = sns.lineplot(
        data=df,
        x="Iteration",
        y="Value",
        hue="Method",
        errorbar=("ci", 95),
        estimator="mean",
        err_style="band",
        palette=palette,
        style="Method",
        dashes=dashes_map
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(df["Method"])], labels[:len(df["Method"])])

    plt.xlabel("Iterations")
    plt.ylabel("Reward" if use_reward(model) else "Reachability")
    # plt.ylim(bottom=-300)
    plt.title(f"HM {model}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,
                f"{model}_convergence_curve.pdf"))
    plt.close()


def plot_robust_lineplot_drone(df, model, output_directory):
    df["Method"] = df["Method"].apply(renamer)
    df = remove_data_drone(df)
    palette = sns.color_palette("tab10")
    # Palette = green, blue, red

    linestyles = [
        (),                  # "-" solid
        (),                  # "-" solid
        (5, 5),              # "--"
        (5, 5),              # "--"
    ]
    palette = ["green", "red", "green", "red"]
    # palette = ["green", "green", "orange", "black"]

    dashes_map = {m: ls for m, ls in zip(df["Method"].unique(), linestyles)}

    plt.figure(figsize=(3.9, 3.9))
    ax = sns.lineplot(
        data=df,
        x="Iteration",
        y="Value",
        hue="Method",
        errorbar=("ci", 95),
        estimator="mean",
        err_style="band",
        palette=palette,
        style="Method",
        dashes=dashes_map
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(df["Method"])], labels[:len(df["Method"])])

    plt.xlabel("Iterations")
    plt.ylabel("Reward" if use_reward(model) else "Reachability")
    # plt.ylim(bottom=-500)
    plt.title(f"HM Drone-2-6-1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,
                f"{model}_convergence_curve.pdf"))
    plt.close()


def plot_robust_moving(df, model, output_directory):
    df["Method"] = df["Method"].apply(renamer)
    df = remove_data_obstacles(df)
    # palette = sns.color_palette("tab10")
    linestyles = [
        (),                  # "-" solid
        (),                  # "-" solid
        (5, 5),              # "--"
        (5, 5),              # "--"
        # (5, 5),              # "--"
    ]
    palette = ["green", "blue", "green", "blue",]

    dashes_map = {m: ls for m, ls in zip(df["Method"].unique(), linestyles)}

    plt.figure(figsize=(3.9, 3.9))
    ax = sns.lineplot(
        data=df,
        x="Iteration",
        y="Value",
        hue="Method",
        errorbar=("ci", 95),
        estimator="mean",
        err_style="band",
        palette=palette,
        style="Method",
        dashes=dashes_map
    )

    plt.xlabel("Iterations")
    plt.ylabel("Reward" if use_reward(model) else "Reachability")
    plt.ylim(bottom=-450)
    plt.title(f"HM Moving Obstacles")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,
                f"{model}_convergence_curve.pdf"))
    plt.close()


def plot_fsc_sizes(df, model, output_directory):
    # aux_df = df[df["Method"] == "FSC Size si-g"]
    # aux_df = pd.concat([aux_df, df[df["Method"] == "FSC Size alergia"]])
    # Include only the Synthesized Worst-Case si-g and Synthesized Worst-Case alergia methods
    aux_df = df[df["Method"] == "Worst-Case SIG"]
    aux_df = pd.concat([aux_df, df[df["Method"] == "Worst-Case Alergia"]])
    print(aux_df["Method"].unique())
    # Make the sizes absolute values
    aux_df["FSC Size"] = aux_df["FSC Size"].abs()
    # aux_df["Method"] = aux_df["Method"].apply(renamer)
    plt.figure(figsize=(6, 6))
    sns.lineplot(
        data=aux_df,
        x="Iteration",
        y="FSC Size",
        hue="Method",
        palette="tab10",
        errorbar=("ci", 95),
        estimator="mean",
        err_style="band",
    )
    plt.xlabel("Iterations")
    plt.ylabel("FSC Size")
    plt.title(f"SIG {model} FSC Sizes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,
                f"{model}_fsc_size.pdf"))
    plt.close()


def read_values_and_times_from_saynt_benchmarks(directory):
    values = {}
    times = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and '_30' in filename:
            file_path = os.path.join(directory, filename)
            model_name = filename.split('_30')[0]
            if model_name not in values:
                values[model_name] = []
                times[model_name] = []
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if "Value =" in line:
                            match = re.search(
                                r'Value = \s*([-+]?\d*\.\d+|\d+)', line)
                            if match:
                                values[model_name].append(
                                    float(match.group(1)))
                            match_time = re.search(
                                r'Time elapsed = \s*([-+]?\d*\.\d+|\d+)', line)
                            if match_time:
                                times[model_name].append(
                                    float(match_time.group(1)))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return values, times


def generate_convergence_curves_joint(benchmark_stats, output_directory, saynt_values, saynt_times, models_to_include=["evade-n17", "intercept-16", "drone-2-8-1"]):
    multiple_data = []
    models_included = []
    os.makedirs(output_directory, exist_ok=True)
    for model in benchmark_stats:
        if model not in models_to_include:
            continue
        else:
            if model not in models_included:
                models_included.append(model)
        stats = benchmark_stats[model]
        records = {
            "RL Values": [],
        }

        for entry in stats:
            if use_reward_single(model):  # use returns
                records["RL Values"].append(
                    ast.literal_eval(entry[1]["returns"]))
            else:  # use reachabilities
                records["RL Values"].append(
                    ast.literal_eval(entry[1]["reach_probs"]))
        if is_negative_single(model):
            for key in records:
                for i in range(len(records[key])):
                    records[key][i] = [-abs(v) for v in records[key][i]]
        max_length = max(len(v) for values in records.values() for v in values)
        data = []
        for label, values_list in records.items():
            for i, values in enumerate(values_list):
                padded = values + [values[-1]] * (max_length - len(values))
                for iteration, value in enumerate(padded):
                    time = (iteration / 41) * 1800
                    data.append({
                        "Iteration": iteration,
                        "Value": value,
                        "Method": label,
                        "Time": time,
                        "Model": model,
                    })
        # Remove decreasing iterations from saynt values (and saynt times as well)
        if model in saynt_values:
            filtered_saynt_values = []
            filtered_saynt_times = []
            last_value = float(
                '-inf') if not is_negative_single(model) else float('inf')
            for v, t in zip(saynt_values[model], saynt_times[model]):
                if (is_negative_single(model) and v <= last_value) or (not is_negative_single(model) and v >= last_value):
                    filtered_saynt_values.append(v)
                    filtered_saynt_times.append(t)
                    last_value = v
            for v, t in zip(filtered_saynt_values, filtered_saynt_times):
                if is_negative_single(model):
                    v = -v
                data.append({
                    "Iteration": None,
                    "Value": v,
                    "Method": "Saynt",
                    "Time": t,
                    "Model": model,
                })

        multiple_data.extend(data)
        # print(data)
    df = pd.DataFrame(multiple_data)
    plt.figure(figsize=(3.9, 3.9))
    sns.lineplot(
        data=df[df["Method"] == "RL Values"],
        x="Time",
        y="Value",
        hue="Model",
        # style="Model",
        palette="tab10",
        errorbar=("ci", 95),
        estimator="mean",
        err_style="band",
        legend="full",
        linestyle="--"
    )
    # SAYNT as crosses (no line) with hue over models as well
    sns.scatterplot(
        data=df[df["Method"] == "Saynt"],
        x="Time",
        y="Value",
        hue="Model",
        # style="Model",
        palette="tab10",
        marker="X",
        s=200,
        legend=False,
    )
    for i, model in enumerate(models_included):
        if model in CONSTANTS_SINGLE_PAYNT_SIG:
            constant_value = CONSTANTS_SINGLE_PAYNT_SIG[model]
            if is_negative_single(model):
                constant_value = -constant_value
            # Použití barvy z palety podle indexu modelu
            palette = sns.color_palette("tab10")
            plt.axhline(
                y=constant_value,
                linestyle='-',
                # Zajistí, že index nezpůsobí chybu
                color=palette[i % len(palette)],
                label=f'{model} Constant Value'
            )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reward" if use_reward_single(
        models_to_include[0]) else "Reachability")
    plt.title(f"Single-POMDP Convergence Curves")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,
                f"joint_convergence_curve.pdf"), bbox_inches='tight')
    plt.close()


def generate_convergence_curves_single(benchmark_stats, output_directory, saynt_values, saynt_times):

    os.makedirs(output_directory, exist_ok=True)
    for model in benchmark_stats:
        stats = benchmark_stats[model]
        records = {
            "RL Values": [],
        }
        for entry in stats:
            if use_reward_single(model):  # use returns
                records["RL Values"].append(
                    ast.literal_eval(entry[1]["returns"]))
            else:  # use reachabilities
                records["RL Values"].append(
                    ast.literal_eval(entry[1]["reach_probs"]))
        if is_negative_single(model):
            for key in records:
                for i in range(len(records[key])):
                    records[key][i] = [-abs(v) for v in records[key][i]]
        max_length = max(len(v) for values in records.values() for v in values)
        data = []
        for label, values_list in records.items():
            for i, values in enumerate(values_list):
                padded = values + [values[-1]] * (max_length - len(values))
                for iteration, value in enumerate(padded):
                    time = 1800 * (iteration / 41)
                    data.append({
                        "Iteration": iteration,
                        "Value": value,
                        "Method": label,
                        "Time": time,
                    })
        # Remove decreasing iterations from saynt values (and saynt times as well)
        if model in saynt_values:
            filtered_saynt_values = []
            filtered_saynt_times = []
            last_value = float(
                '-inf') if not is_negative_single(model) else float('inf')
            for v, t in zip(saynt_values[model], saynt_times[model]):
                if (is_negative_single(model) and v <= last_value) or (not is_negative_single(model) and v >= last_value):
                    filtered_saynt_values.append(v)
                    filtered_saynt_times.append(t)
                    last_value = v
            for v, t in zip(filtered_saynt_values, filtered_saynt_times):
                if is_negative_single(model):
                    v = -v
                data.append({
                    "Iteration": None,
                    "Value": v,
                    "Method": "Saynt",
                    "Time": t,
                })
        # print(data)
        df = pd.DataFrame(data)
        plt.figure(figsize=(3, 3))
        sns.lineplot(
            data=df[df["Method"] == "RL Values"],
            x="Time",
            y="Value",
            color="orange",
            errorbar=("ci", 95),
            estimator="mean",
            err_style="band",
            label="RL Values",
            linestyle="--"
        )

        # Saynt jako křížky (bez křivky)
        sns.scatterplot(
            data=df[df["Method"] == "Saynt"],
            x="Time",
            y="Value",
            color="red",
            marker="X",
            s=100,
            label="SAYNT",
        )
        # Add constant values for models if available
        if model in CONSTANTS_SINGLE_PAYNT_SIG:
            constant_value = CONSTANTS_SINGLE_PAYNT_SIG[model]
            if is_negative_single(model):
                constant_value = -constant_value
            plt.axhline(y=constant_value, color='blue',
                        linestyle='-', label='Verified SIG')
        if model in CONSTANTS_SINGLE_PAYNT_ALERGIA:
            constant_value = CONSTANTS_SINGLE_PAYNT_ALERGIA[model]
            if is_negative_single(model):
                constant_value = -constant_value
            plt.axhline(y=constant_value, color='green',
                        linestyle='-', label='Verified Alergia')

        plt.xlabel("Time (seconds)")
        plt.ylabel("Reward" if use_reward_single(model) else "Reachability")
        plt.title(f"{model}")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()  # Zobrazí legendu
        plt.savefig(os.path.join(output_directory,
                    f"{model}_convergence_curve.pdf"))
        plt.close()


def generate_convergence_robust():

    base_directory = "./models/models_table_camera_ready"
    output_directory = "./convergence_curves"
    benchmark_stats = load_all_benchmark_directories(base_directory)
    # generate_convergence_curves(benchmark_stats, output_directory)
    # base_directory = "./models/models_gru_experiment"
    # output_directory = "./convergence_curves_gru_experiment"
    # benchmark_stats = load_all_benchmark_directories(base_directory)
    generate_convergence_curves(
        benchmark_stats, output_directory, gru_extracted=False)


def generate_convergence_single():
    base_directory = "./models/models_single_pomdp"
    output_directory = "./convergence_curves_single"
    if not os.path.exists("./results_saynt"):
        print(
            "Directory ./results_saynt does not exist. Please run SAYNT benchmarks first.")
        saynt_values, saynt_times = {}, {}
    else:
        saynt_values, saynt_times = read_values_and_times_from_saynt_benchmarks(
            "./results_saynt")
    benchmark_stats = load_all_benchmark_directories(
        base_directory, training_include=True)
    generate_convergence_curves_single(
        benchmark_stats, output_directory, saynt_values, saynt_times)
    generate_convergence_curves_joint(
        benchmark_stats, "./convergence_curves_joint", saynt_values, saynt_times)


if __name__ == "__main__":
    generate_convergence_robust()
    generate_convergence_single()
