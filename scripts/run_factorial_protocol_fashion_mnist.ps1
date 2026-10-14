# run_factorial_protocol_fashion_mnist.ps1
# Runs the main scientific protocol for Fashion-MNIST (3 topologies x 4 latent dimensions).
# Requires internet access for the first run (downloads Fashion-MNIST).
# On CPU this takes a long time: 12 unsupervised trainings + 1 supervised training.

Write-Host "=== VAE Project: Fashion-MNIST Factorial Protocol ===" -ForegroundColor Cyan
Write-Host "Grid: small/medium/deep x 2D/8D/16D/32D (12 unsupervised configurations)" -ForegroundColor Yellow
Write-Host "Artifacts: artifacts/fashion_mnist/unsupervised/factorial_protocol/run_*/" -ForegroundColor White
Write-Host "To resume an interrupted run, add: --resume <run_directory>" -ForegroundColor White
Write-Host ""

python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_fashion_mnist.yaml

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Fashion-MNIST factorial protocol finished. Artifacts saved to: artifacts/" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Protocol FAILED. See protocol_summary.md and logs/protocol.log in the run directory." -ForegroundColor Red
    exit 1
}
