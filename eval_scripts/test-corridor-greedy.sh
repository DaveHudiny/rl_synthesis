#!/bin/bash
SUBFOLDER="$1"
if [ -z "$SUBFOLDER" ]; then
	echo "Usage: $0 <results_subfolder>"
	exit 1
fi

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/test-corridor-greedy-01.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.1 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/test-corridor-greedy-01.log

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/test-corridor-greedy-02.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.2 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/test-corridor-greedy-02.log

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/test-corridor-greedy-005.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.05 --number-of-evaluations 3 --log-file results/"$SUBFOLDER"/test-corridor-greedy-005.log