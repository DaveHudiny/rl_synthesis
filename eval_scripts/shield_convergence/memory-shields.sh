#!/bin/bash


SUBFOLDER="$1"
ARG2="$2"
ARG3="$3"

if [ -z "$SUBFOLDER" ]; then
	echo "Usage: $0 <results_subfolder> [deterministic] [int]"
	exit 1
fi

DET_FLAG=""
MEMORY_FLAG=""

# Check ARG2 and ARG3 for deterministic or number
for ARG in "$ARG2" "$ARG3"; do
	if [ "$ARG" = "deterministic" ]; then
		DET_FLAG="--deterministic-agent"
	elif [[ "$ARG" =~ ^[0-9]+$ ]]; then
		MEMORY_FLAG="--shield-memory $ARG"
	fi
done

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log $DET_FLAG $MEMORY_FLAG --shield-file-name convergence --only-self-constructing

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/slippy-drone-greedy-02-memory.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-4000 --nu 0.2 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/slippy-drone-greedy-02-memory.log $DET_FLAG $MEMORY_FLAG --shield-file-name convergence --only-self-constructing

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/test-corridor-greedy-02-memory.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.2 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/test-corridor-greedy-02-memory.log $DET_FLAG $MEMORY_FLAG --shield-file-name convergence --only-self-constructing