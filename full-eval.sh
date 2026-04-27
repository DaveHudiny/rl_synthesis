#!/usr/bin/env bash

set -e pipefail

./eval_scripts/full_eval/construct_shields.sh

if [ -d results/eval-shield-review ] && [ ! -d results/eval-shield-full ]; then
	mkdir -p results/eval-shield-full
	cp -a results/eval-shield-review/. results/eval-shield-full/
fi

./eval_scripts/full_eval/eval_shields.sh

./eval_scripts/full_eval/runtimes.sh

mkdir -p results/temp-files
python3 eval_scripts/create_table_data.py results/eval-shield-full/evaluation_results.csv > results/temp-files/full-table-data.tex

cd results/temp-files
pdflatex full-table-data.tex 1>/dev/null
mv full-table-data.pdf ../full-table.pdf
cd -

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review > results/review-calls-per-second-table.tex

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review --data-type time > results/review-runtime-table.tex

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review --data-type memory > results/review-memory-table.tex

printf '\nARTIFACT REVIEW EVALUATION COMPLETE! Check the generated tables located at results/review-table.pdf, results/review-calls-per-second-table.tex, results/review-runtime-table.tex, and results/review-memory-table.tex\n\n'