python3 eval_shield.py --results-folder results/eval-shield-full --shield standard&

python3 eval_shield.py --results-folder results/eval-shield-full --shield delta&

python3 eval_shield.py --results-folder results/eval-shield-full --shield offline --auto-offline-shield-name&

wait

python3 eval_shield.py --results-folder results/eval-shield-full --shield optimistic&

python3 eval_shield.py --results-folder results/eval-shield-full --shield pessimistic&

python3 eval_shield.py --results-folder results/eval-shield-full --shield online&

python3 eval_shield.py --results-folder results/eval-shield-full --shield offline --shield-memory 1 --auto-offline-shield-name&

wait