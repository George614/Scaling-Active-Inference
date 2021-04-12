#!/bin/bash
# run PPL training/experiments given seed

start_seed=11
let end_seed=start_seed+14
for seed in $(seq $start_seed $end_seed)
do
echo seed $seed
python train_ppl_model.py --seed=$seed
done
echo All done