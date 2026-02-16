#!/bin/bash

echo "Running DAPT..."
python training/run_dapt.py > experiments/logs/dapt_training.log

echo "Running Alignment..."
python training/run_alignment.py > experiments/logs/alignment_training.log

echo "Evaluating..."
python training/statistical_evaluation.py

echo "Generating Visualizations..."
python training/visualize_embeddings.py

echo "Pipeline complete."
