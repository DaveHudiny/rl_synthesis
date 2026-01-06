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


python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 4096 --num-parallel-environments 16 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log --shield-file-name dpm-greedy-memory-convergence-1st --only-self-constructing --shield-memory 0 $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 4096 --num-parallel-environments 16 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log --shield-file-name dpm-greedy-memory-convergence-1st --only-self-constructing --shield-memory 1 $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 4096 --num-parallel-environments 16 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log --shield-file-name dpm-greedy-memory-convergence-1st --only-self-constructing --shield-memory 2 $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 4096 --num-parallel-environments 16 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log --shield-file-name dpm-greedy-memory-convergence-1st --only-self-constructing --shield-memory 3 $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 4096 --num-parallel-environments 16 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log --shield-file-name dpm-greedy-memory-convergence-1st --only-self-constructing --shield-memory 4 $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 4096 --num-parallel-environments 16 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log --shield-file-name dpm-greedy-memory-convergence-1st --only-self-constructing --shield-memory 5 $DET_FLAG

python3 run_benchmark.py --results-file results/"$SUBFOLDER"/dpm-greedy-02-memory.csv --model-path models/shielding_test/dpm  --episode-length 50 --num-environments 4096 --num-parallel-environments 16 --agent greedy-iter-1000 --nu 0.2 --goal-rew 0.0 --number-of-evaluations 1 --log-file results/"$SUBFOLDER"/dpm-greedy-02-memory.log --shield-file-name dpm-greedy-memory-convergence-1st --only-self-constructing --shield-memory 6 $DET_FLAG
