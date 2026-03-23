#!/bin/bash

# Define arrays for parameters
SWITCH_INTERVALS=(25 250 500)
MUTATION_PROBS=(0.7)  # 0.8 0.1
N_AGENTS=(5 15)  #  30 45

# The combinations of selection and multi_level_selection
SELECTIONS=("individual-local" "individual-global")
MLS_FLAGS=("True" "False")

for n_ag in "${N_AGENTS[@]}"; do
    for mp in "${MUTATION_PROBS[@]}"; do
        for si in "${SWITCH_INTERVALS[@]}"; do
            for i in "${!SELECTIONS[@]}"; do
                sel="${SELECTIONS[$i]}"
                mls="${MLS_FLAGS[$i]}"
                
                echo "Submitting: n_agents=$n_ag, mutation_prob=$mp, switch_interval=$si, selection=$sel"
                
                uv run --extra cu124 python submit_ea.py \
                    --gpu \
                    --replicates 2000 \
                    --generations 3000 \
                    --switch_interval "$si" \
                    --selection "$sel" \
                    --multi_level_selection "$mls" \
                    --mutation_prob "$mp" \
                    --n_agents "$n_ag"
            done
        done
    done
done
