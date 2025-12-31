#!/bin/bash

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

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/test-corridor-uniform-01.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-100 --nu 0.1 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/test-corridor-uniform-01.log --uniform-random-agent $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/test-corridor-uniform-02.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-100 --nu 0.2 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/test-corridor-uniform-02.log --uniform-random-agent $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/test-corridor-uniform-005.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-100 --nu 0.05 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/test-corridor-uniform-005.log --uniform-random-agent $DET_FLAG