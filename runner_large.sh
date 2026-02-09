#!/bin/bash
# This is the main script that runs all experiments.

models_dir="models/models_robust"

seeds=(12345 23456 34567 45678 56789 67890 78901 89012 90123 01234)

# Generate data for robust experiments.
for seed in "${seeds[@]}"; do
    echo "Evaluating models with seed $seed"
    for extraction_method in "si-g" "alergia"; do
        mkdir -p "results_robust_large/${extraction_method}_seed${seed}"
    done

    for extraction_method in "si-g" "alergia"; do
        for model in $(ls $models_dir); do
            # if model is not directory, skip
            if [ ! -d "$models_dir/$model" ]; then
                echo "Skipping $model as it is not a directory"
                continue
            fi
            echo "Evaluating $model with extraction method $extraction_method and seed $seed"
            timeout 3600 prerequisites/venv/bin/python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
                --batched-vec-storm \
                --geometric-batched-vec-storm \
                --extraction-method $extraction_method \
                --seed $seed \
                > "results_robust_large/${extraction_method}_seed${seed}/${model}_output.txt" 2>&1
            echo "Finished evaluating $model with extraction method $extraction_method and seed $seed"
        done
    done
done

models_dir="models/models_single_pomdp"
seeds=(12345 23456 34567 45678 56789 67890 78901 89012 90123 01234)

# Generate data for single POMDP learning experiments.
for seed in "${seeds[@]}"; do
    echo "Evaluating models with seed $seed"
    for extraction_method in "si-g" "alergia"; do
        mkdir -p "results_single/${extraction_method}_seed${seed}"
    done

    for extraction_method in "si-g" "alergia"; do
        for model in $(ls $models_dir); do
            # if model is not directory, skip
            if [ ! -d "$models_dir/$model" ]; then
                echo "Skipping $model as it is not a directory"
                continue
            fi
            echo "Evaluating $model with extraction method $extraction_method and seed $seed"
            timeout 3600 prerequisites/venv/bin/python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
                --batched-vec-storm \
                --geometric-batched-vec-storm \
                --extraction-method $extraction_method \
                --seed $seed \
                > "results_single/${extraction_method}_seed${seed}/${model}_output.txt" 2>&1
            echo "Finished evaluating $model with extraction method $extraction_method and seed $seed"
        done
    done
done

models_dir="models/models_robust"
seeds=(12345 23456 34567 45678 56789)

# Generate data to compare robust learning vs random POMDP learning. 
for seed in "${seeds[@]}"; do
    echo "Evaluating models with seed $seed"
    for extraction_method in "si-g" "alergia"; do
        mkdir -p "results_robust_experiments/${extraction_method}_seed${seed}"
    done
    for extraction_method in "si-g" "alergia"; do
        for model in $(ls $models_dir); do
            # If model is not drone-2-6-1, skip
            if [ "$model" != "drone-2-6-1" ]; then
                echo "Skipping $model as it is not drone-2-6-1"
                continue
            fi
            # if model is not directory, skip
            if [ ! -d "$models_dir/$model" ]; then
                echo "Skipping $model as it is not a directory"
                continue
            fi
            echo "Evaluating $model with extraction method $extraction_method and seed $seed"
            timeout 3600 prerequisites/venv/bin/python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
                --batched-vec-storm \
                --geometric-batched-vec-storm \
                --extraction-method $extraction_method \
                --seed $seed \
                --without-extraction \
                > "results_robust_experiments/${extraction_method}_seed${seed}/${model}_output.txt" 2>&1
            echo "Finished evaluating $model with extraction method $extraction_method and seed $seed"
        done
    done
done

models_dir="models/models_robust"
seeds=(12345 23456 34567 45678 56789)
# Generate data to compare the GRU extraction vs si-g.
for seed in "${seeds[@]}"; do
    echo "Evaluating models with seed $seed"
    for extraction_method in "si-g"; do
        mkdir -p "results_gru_comparison/${extraction_method}_seed${seed}"
    done
    for extraction_method in "si-g"; do
        for model in $(ls $models_dir); do
            # If model is not drone-2-6-1, skip
            if [ "$model" != "avoid-large" ]; then
                echo "Skipping $model as it is not avoid-large"
                continue
            fi
            # if model is not directory, skip
            if [ ! -d "$models_dir/$model" ]; then
                echo "Skipping $model as it is not a directory"
                continue
            fi
            echo "Evaluating $model with extraction method $extraction_method and seed $seed"
            # The timeout is 2 hours for GRU experiments, since we now use two separate extractions to collect data.
            timeout 10000 prerequisites/venv/bin/python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
                --batched-vec-storm \
                --geometric-batched-vec-storm \
                --extraction-method $extraction_method \
                --seed $seed \
                --with-gru \
                > "results_gru_comparison/${extraction_method}_seed${seed}/${model}_output.txt" 2>&1
            echo "Finished evaluating $model with extraction method $extraction_method and seed $seed"
        done
    done
done

