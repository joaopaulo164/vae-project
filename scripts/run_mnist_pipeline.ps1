# run_mnist_pipeline.ps1
# Runs the complete MNIST unsupervised VAE pipeline.
# Requires internet access for the first run (downloads MNIST).

Write-Host "=== VAE Project: MNIST Pipeline ===" -ForegroundColor Cyan
Write-Host "Dataset: MNIST | Experiment: Unsupervised VAE" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will:" -ForegroundColor White
Write-Host "  1. Download MNIST (first run only)" -ForegroundColor White
Write-Host "  2. Train VAE with medium topology (latent_dim=16)" -ForegroundColor White
Write-Host "  3. Generate projections (PCA, UMAP, t-SNE)" -ForegroundColor White
Write-Host "  4. Save all artifacts to: artifacts/mnist/unsupervised/latent_16/" -ForegroundColor White
Write-Host ""

python -m vae_project.cli run-pipeline --config configs/mnist_unsupervised.yaml

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "MNIST pipeline complete. Artifacts saved to: artifacts/" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Pipeline FAILED. Check logs in the artifacts directory." -ForegroundColor Red
    exit 1
}
