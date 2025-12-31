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

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-1000 --nu 0.05 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-1000 --nu 0.01 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.05 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"//model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.01 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"//model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.05 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.01 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-4000 --nu 0.05 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-4000 --nu 0.2 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-4000 --nu 0.01 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.05 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.2 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-1000 --nu 0.01 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.05 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.2 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/slippy-drone  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-1000 --nu 0.01 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.1 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.2 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent greedy-iter-100 --nu 0.05 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-4000 --nu 0.1 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-4000 --nu 0.2 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent safe-iter-4000 --nu 0.05 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-100 --nu 0.1 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-100 --nu 0.2 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/model-check-results.csv --model-path models/shielding_test/test-corridor  --episode-length 50 --num-environments 1024 --num-parallel-environments 256 --agent uniform-iter-100 --nu 0.05 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/model-check-results.log --uniform-random-agent --model-checking-eval $DET_FLAG