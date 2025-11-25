#!/bin/bash

for seed in 0 1 2; do
    for i in 0 1 2 3 4 5; do
        python meta_memory/scripts/eval.py --sequence_id $i 
    done
done
