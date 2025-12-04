#!/bin/bash

# Quick test script - runs a fast test with minimal samples
# Usage: ./quick_test.sh [dataset_name]

DATASET="${1:-poisson}"

echo "Running quick test on $DATASET dataset..."
echo ""

python test_suite.py \
    --dataset "$DATASET" \
    --grid_size 64 \
    --modes 12 \
    --batch_size 10 \
    --n_train 20 \
    --n_test 10 \
    --n_epochs 5 \
    --lr 0.001 \
    --seed 42 \
    --output_dir results

echo ""
echo "Quick test completed!"

