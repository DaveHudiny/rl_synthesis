#!/bin/bash
SUBFOLDER="$1"
if [ -z "$SUBFOLDER" ]; then
	echo "Usage: $0 <results_subfolder>"
	exit 1
fi

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-safe-005.csv --model-path models/shielding_test/dpm  --episode-length 100 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.05 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/dpm-safe-005.log

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-safe-02.csv --model-path models/shielding_test/dpm  --episode-length 100 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.2 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/dpm-safe-02.log

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-safe-001.csv --model-path models/shielding_test/dpm  --episode-length 100 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.01 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/dpm-safe-001.log