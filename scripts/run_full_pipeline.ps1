Write-Host "Running DAPT..." -ForegroundColor Green
python training/run_dapt.py | Out-File -FilePath experiments/logs/dapt_training.log

Write-Host "Running Alignment..." -ForegroundColor Green
python training/run_alignment.py | Out-File -FilePath experiments/logs/alignment_training.log

Write-Host "Evaluating..." -ForegroundColor Green
python training/statistical_evaluation.py

Write-Host "Generating Visualizations..." -ForegroundColor Green
python training/visualize_embeddings.py

Write-Host "Pipeline complete." -ForegroundColor Cyan
