#!/usr/bin/env bash

set -e pipefail

./eval_scripts/artifact_review-single-core/construct_shields.sh

./eval_scripts/artifact_review-single-core/eval_shields.sh

./eval_scripts/artifact_review-single-core/runtimes.sh

mkdir -p results/temp-files
python3 eval_scripts/create_table_data.py results/eval-shield-review/evaluation_results.csv --artifact-review > results/temp-files/review-table-data.tex

cd results/temp-files
pdflatex review-table-data.tex 1>/dev/null
mv review-table-data.pdf ../review-table.pdf
cd -

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review > results/review-calls-per-second-table.tex

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review --data-type time > results/review-runtime-table.tex

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review --data-type memory > results/review-memory-table.tex

printf '\nARTIFACT REVIEW EVALUATION COMPLETE! Check the generated tables located at results/review-table.pdf, results/review-calls-per-second-table.tex, results/review-runtime-table.tex, and results/review-memory-table.tex\n\n'