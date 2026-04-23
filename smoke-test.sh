./eval_scripts/smoke_test/construct_shields.sh

./eval_scripts/smoke_test/eval_shields.sh

python3 eval_scripts/create_table_data.py results/eval-shield-smoke-test/evaluation_results.csv --artifact-review > results/temp-files/smoke-test-table-data.tex

cd results/temp-files && pdflatex smoke-test-table-data.tex && mv smoke-test-table-data.pdf ../smoke-test-table.pdf && cd -