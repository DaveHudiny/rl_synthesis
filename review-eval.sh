./eval-scripts/artifact_review/construct_shields.sh

./eval-scripts/artifact_review/eval_shields.sh

./eval-scripts/artifact_review/runtimes.sh

python3 eval_scripts/create_table_data.py results/eval-shield-review/evaluation_results.csv --artifact-review > results/temp-files/review-table-data.tex

cd results/temp-files && pdflatex review-table-data.tex && mv review-table-data.pdf ../review-table.pdf && cd -

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review > results/review-calls-per-second-table.tex

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review --data-type time > results/review-runtime-table.tex

python3 eval_scripts/create_runtimes_table.py results/runtimes-review/evaluation_results.csv --artifact-review --data-type memory > results/review-memory-table.tex