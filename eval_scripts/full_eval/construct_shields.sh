python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-model corridor --artifact-review&

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-model drone --artifact-review&

wait

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-model dpm --artifact-review&

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-model drone-b&

wait

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-memory 1 --shield-model dpm --artifact-review&

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-memory 1 --shield-model corridor --artifact-review&

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-memory 1 --shield-model drone --artifact-review&

wait

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-memory 1 --shield-model drone-b --shield-construction-agent greedy --artifact-review&

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-memory 1 --shield-model drone-b --shield-construction-agent safe --artifact-review&

wait

python3 construct_shield_for_eval.py --results-folder results/shield-construction-review --shield-memory 1 --shield-model drone-b --shield-construction-agent random --artifact-review&

wait