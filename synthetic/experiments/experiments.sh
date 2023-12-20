#!/bin/bash

seed=1

# Echo the seed for confirmation
echo "Running experiment with seed: $seed"

# This script runs each job in parallel, if you want to run them sequentially, remove the & at the end of this script for less resource-intensive runs. 
for setting in redundancy uniqueness0 uniqueness1 synergy mix1 mix2 mix3 mix4 mix5 mix6 # synthetic1 synthetic2 synthetic3 synthetic4 synthetic5
do
    (
    echo ${setting}
    mkdir -p synthetic/experiments/${setting}

    # late fusion-avg
    echo additive
    python synthetic/additive.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_additive_best.pt --setting ${setting} --out_dir synthetic/experiments/${setting} --seed ${seed} > synthetic/experiments/${setting}/${setting}_additive_${seed}.txt

    ) & 
done