Write-Host "Recomputing evaluation..." -ForegroundColor Green
python training/statistical_evaluation.py

Write-Host "Rebuilding UMAP figures..." -ForegroundColor Green
python training/visualize_embeddings.py

Write-Host "Reproduction complete." -ForegroundColor Cyan
