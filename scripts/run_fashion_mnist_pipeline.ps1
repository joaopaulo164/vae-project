# run_fashion_mnist_pipeline.ps1
# Runs the complete Fashion-MNIST unsupervised VAE pipeline.

Write-Host "=== VAE Project: Fashion-MNIST Pipeline ===" -ForegroundColor Cyan
Write-Host "Dataset: Fashion-MNIST | Experiment: Unsupervised VAE" -ForegroundColor Yellow
Write-Host ""

python -m vae_project.cli run-pipeline --config configs/fashion_mnist_unsupervised.yaml

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Fashion-MNIST pipeline complete. Artifacts saved to: artifacts/" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Pipeline FAILED. Check logs in the artifacts directory." -ForegroundColor Red
    exit 1
}
