#!/bin/bash

# Usage examples:
#   bash dpm-greedy.sh <results_subfolder>
#   bash dpm-greedy.sh <results_subfolder> deterministic

SUBFOLDER="$1"
DETERMINISTIC="$2"
if [ -z "$SUBFOLDER" ]; then
	echo "Usage: $0 <results_subfolder> [deterministic]"
	exit 1
fi

DET_FLAG=""
if [ "$DETERMINISTIC" = "deterministic" ]; then
	DET_FLAG="--deterministic-agent"
fi



python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-005.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-1000 --nu 0.05 --goal-rew 0.0 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/dpm-greedy-005.log $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/dpm-greedy-02.log $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-001.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-1000 --nu 0.01 --goal-rew 0.0 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/dpm-greedy-001.log $DET_FLAG