#!/usr/bin/env bash

set -e pipefail

python3 eval_convergence.py --results-folder results/convergence --shield-model dpm --construct-shields --shield-name convergence&

python3 eval_convergence.py --results-folder results/convergence --shield-model drone-b --construct-shields --shield-name convergence&

wait

python3 eval_convergence.py --results-folder results/convergence --shield-model corridor --construct-shields --shield-name convergence&

python3 eval_convergence.py --results-folder results/convergence --shield-model drone --construct-shields --shield-name convergence&

wait

python3 eval_convergence.py --results-folder results/convergence --shield-name convergence

python3 eval_scripts/create_convergence_plot.py --csv-file results/convergence/convergence.csv --output results/convergence.pdf
