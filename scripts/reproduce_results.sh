#!/bin/bash

echo "Recomputing evaluation..."
python training/statistical_evaluation.py

echo "Rebuilding UMAP figures..."
python training/visualize_embeddings.py

echo "Reproduction complete."
