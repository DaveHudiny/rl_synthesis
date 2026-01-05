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

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/slippy-drone-uniform-005.csv --model-path models/shielding_test/slippy-drone  --episode-length 100 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.05 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/slippy-drone-uniform-005.log --uniform-random-agent $DET_FLAG --shield-file-name ep-100

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/slippy-drone-uniform-02.csv --model-path models/shielding_test/slippy-drone  --episode-length 100 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.2 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/slippy-drone-uniform-02.log --uniform-random-agent $DET_FLAG --shield-file-name ep-100

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/slippy-drone-uniform-001.csv --model-path models/shielding_test/slippy-drone  --episode-length 100 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.01 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/slippy-drone-uniform-001.log --uniform-random-agent $DET_FLAG --shield-file-name ep-100