#!/bin/bash

SUBFOLDER="$1"
if [ -z "$SUBFOLDER" ]; then
	echo "Usage: $0 <results_subfolder>"
	exit 1
fi

nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu001-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent greedy --shield-nu 0.01
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent greedy --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent greedy --shield-nu 0.2

nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu001-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent safe --shield-nu 0.01
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent safe --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent safe --shield-nu 0.2

nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu001-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent random --shield-nu 0.01
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent random --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model dpm --shield-construction-agent random --shield-nu 0.2


nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent greedy --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu01-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent greedy --shield-nu 0.1
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent greedy --shield-nu 0.2

nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent safe --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu01-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent safe --shield-nu 0.1
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent safe --shield-nu 0.2

nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent random --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu01-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent random --shield-nu 0.1
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model corridor --shield-construction-agent random --shield-nu 0.2


nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu001-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent greedy --shield-nu 0.01
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent greedy --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path greedy-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent greedy --shield-nu 0.2

nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu001-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent safe --shield-nu 0.01
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent safe --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path safe-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent safe --shield-nu 0.2

nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu001-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent random --shield-nu 0.01
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu005-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent random --shield-nu 0.05
nohup python3 eval_shield.py --results-folder "$SUBFOLDER" --shield offline --shield-path random-nu02-mem0--eval-1-iter-final-shield.pickle --shield-model drone --shield-construction-agent random --shield-nu 0.2