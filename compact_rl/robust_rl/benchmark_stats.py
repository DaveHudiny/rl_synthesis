

from os import path
import json

import time

class BenchmarkStats:
    def __init__(self, fsc_size=3, num_training_steps_per_iteration=50, batched_vec_storm=True,
                 extraction_type: str = "alergia", lstm_width=32, without_extraction=False,
                 geometric_batched_vec_storm=False, periodic_restarts=False,
                 period_between_worst_case_evaluation=5, nr_pomdps_in_family=1,
                 seed = 42):
        self.nr_pomdps_in_family = nr_pomdps_in_family
        self.fsc_size = fsc_size
        self.num_training_steps_per_iteration = num_training_steps_per_iteration
        self.rl_performance_single_pomdp = []
        self.rl_performance_single_pomdp_reachability = []
        self.extracted_fsc_return = []
        self.extracted_fsc_reachability = []
        self.family_performance = []
        self.available_nodes_in_fsc = []
        self.worst_case_pomdp_values_simulated_rl = []
        self.worst_case_pomdp_values_simulated_fsc = []
        self.was_worst_case_same_as_simulated = []
        self.worst_case_index_rl = []
        self.worst_case_index_fsc = []
        self.worst_case_index_verif = []
        self.number_of_training_trajectories = []
        self.environment_type = "batched_vec_storm" if batched_vec_storm else "vec_storm"
        self.extraction_type = extraction_type
        self.lstm_width = lstm_width
        self.extraction_less = without_extraction
        self.geometric_batched_vec_storm = geometric_batched_vec_storm
        self.periodic_restarts = periodic_restarts
        self.worst_cases_on_subset_rl = []
        self.worst_cases_on_subset_fsc = []
        self.worst_cases_reachability_rl = []
        self.worst_cases_reachability_fsc = []
        self.period_between_worst_case_evaluation = period_between_worst_case_evaluation
        self.shrink_and_perturb_activated = []
        self.nr_of_clusters = []

        self.lstm_extracted_return = []
        self.lstm_extracted_reachability = []
        self.seed = seed

        self.initialize_time_metrics()

    def initialize_time_metrics(self):
        self.initial_time = time.time()
        self.rl_training_times = []
        self.fsc_extraction_times = []
        self.family_performance_times = []
        self.worst_case_evaluation_times_rl = []
        self.worst_case_evaluation_times_fsc = []
        self.lstm_extracted_result_times = []
        self.cluster_evaluation_times = []

    def add_rl_performance(self, performance: float):
        self.rl_performance_single_pomdp.append(performance)
        self.rl_training_times.append(time.time() - self.initial_time)

    def add_rl_performance_reachability(self, performance: float):
        self.rl_performance_single_pomdp_reachability.append(performance)

    def add_extracted_fsc_performance(self, performance):
        self.extracted_fsc_return.append(performance)
        self.fsc_extraction_times.append(time.time() - self.initial_time)

    def add_extracted_fsc_reachability(self, performance):
        self.extracted_fsc_reachability.append(performance)

    def add_family_performance(self, performance):
        self.family_performance.append(performance)
        self.family_performance_times.append(time.time() - self.initial_time)

    def add_worst_case_pomdp_values_simulated_rl(self, value):
        self.worst_case_pomdp_values_simulated_rl.append(value)
        self.worst_case_evaluation_times_rl.append(time.time() - self.initial_time)

    def add_worst_case_pomdp_values_simulated_fsc(self, value):
        self.worst_case_pomdp_values_simulated_fsc.append(value)
        self.worst_case_evaluation_times_fsc.append(time.time() - self.initial_time)

    def add_worst_case_same_as_simulated(self, was_same: bool):
        self.was_worst_case_same_as_simulated.append(was_same)

    def add_worst_case_assignments(self, rl_assignment: str, fsc_assignment: str, verif_assignment: str):
        self.worst_case_index_rl.append(rl_assignment)
        self.worst_case_index_fsc.append(fsc_assignment)
        self.worst_case_index_verif.append(verif_assignment)

    def add_number_of_training_trajectories(self, number_of_trajectories: int):
        self.number_of_training_trajectories.append(number_of_trajectories)

    def add_lstm_extracted_results(self, lstm_extracted_reachability: float, lstm_extracted_return: float):
        self.lstm_extracted_reachability.append(lstm_extracted_reachability)
        self.lstm_extracted_return.append(lstm_extracted_return)

    def add_nr_clusters(self, nr_clusters: int):
        self.nr_of_clusters.append(nr_clusters)
        self.cluster_evaluation_times.append(time.time() - self.initial_time)

    def save_stats(self, path):

        benchmark_stats = self
        stats = {
            "number_of_pomdps_in_family": str(benchmark_stats.nr_pomdps_in_family),
            "num_training_steps_per_iteration": str(benchmark_stats.num_training_steps_per_iteration),
            "average_rl_return_subset_simulated": str(benchmark_stats.rl_performance_single_pomdp),
            "average_rl_reachability_subset_simulated": str(benchmark_stats.rl_performance_single_pomdp_reachability),
            "average_extracted_fsc_return_subset_simulated": str(benchmark_stats.extracted_fsc_return),
            "average_extracted_fsc_reachability_subset_simulated": str(benchmark_stats.extracted_fsc_reachability),
            "family_performance": str(benchmark_stats.family_performance),
            "available_nodes_in_fsc": str(benchmark_stats.available_nodes_in_fsc),
            "worst_case_pomdp_values_simulated_rl": str(benchmark_stats.worst_case_pomdp_values_simulated_rl),
            "worst_case_pomdp_values_simulated_fsc": str(benchmark_stats.worst_case_pomdp_values_simulated_fsc),
            "was_worst_case_same_as_simulated": str(benchmark_stats.was_worst_case_same_as_simulated),
            "worst_case_index_rl": str(benchmark_stats.worst_case_index_rl),
            "worst_case_index_fsc": str(benchmark_stats.worst_case_index_fsc),
            "worst_case_index_paynt": str(benchmark_stats.worst_case_index_verif),
            "environment_type": benchmark_stats.environment_type,
            "number_of_extraction_trajectories": str(benchmark_stats.number_of_training_trajectories),
            "extraction_type": benchmark_stats.extraction_type,
            "lstm_width": benchmark_stats.lstm_width,
            "extraction_less": benchmark_stats.extraction_less,
            "geometric_batched_vec_storm": benchmark_stats.geometric_batched_vec_storm,
            "periodic_restarts": benchmark_stats.periodic_restarts,
            "worst_case_on_subset_rl": str(benchmark_stats.worst_cases_on_subset_rl),
            "worst_case_on_subset_fsc": str(benchmark_stats.worst_cases_on_subset_fsc),
            "worst_cases_reachability_rl": str(benchmark_stats.worst_cases_reachability_rl),
            "worst_cases_reachability_fsc": str(benchmark_stats.worst_cases_reachability_fsc),
            "period_between_worst_case_evaluation": str(benchmark_stats.period_between_worst_case_evaluation),
            "shrink_and_perturb_activated": str(benchmark_stats.shrink_and_perturb_activated),
            "nr_of_clusters": str(benchmark_stats.nr_of_clusters),
            "lstm_extracted_reachability": str(benchmark_stats.lstm_extracted_reachability),
            "lstm_extracted_return": str(benchmark_stats.lstm_extracted_return),
            "seed": str(benchmark_stats.seed),
            "initial_time": str(benchmark_stats.initial_time),
            "rl_training_times": str(benchmark_stats.rl_training_times),
            "fsc_extraction_times": str(benchmark_stats.fsc_extraction_times),
            "family_performance_times": str(benchmark_stats.family_performance_times),
            "worst_case_evaluation_times_rl": str(benchmark_stats.worst_case_evaluation_times_rl),
            "worst_case_evaluation_times_fsc": str(benchmark_stats.worst_case_evaluation_times_fsc),
            "lstm_extracted_result_times": str(benchmark_stats.lstm_extracted_result_times),
            "cluster_evaluation_times": str(benchmark_stats.cluster_evaluation_times)
        }
        with open(path, 'w') as f:
            json.dump(stats, f, indent=4)
