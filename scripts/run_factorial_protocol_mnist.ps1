# run_factorial_protocol_mnist.ps1
# Runs the main scientific protocol for MNIST (3 topologies x 5 latent dimensions).
# Requires internet access for the first run (downloads MNIST).
# On CPU this takes a long time: 15 unsupervised trainings + 1 supervised training.

Write-Host "=== VAE Project: MNIST Factorial Protocol ===" -ForegroundColor Cyan
Write-Host "Grid: small/medium/deep x 2D/4D/8D/16D/32D (15 unsupervised configurations)" -ForegroundColor Yellow
Write-Host "Artifacts: artifacts/mnist/unsupervised/factorial_protocol/run_*/" -ForegroundColor White
Write-Host "To resume an interrupted run, add: --resume <run_directory>" -ForegroundColor White
Write-Host ""

python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "MNIST factorial protocol finished. Artifacts saved to: artifacts/" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Protocol FAILED. See protocol_summary.md and logs/protocol.log in the run directory." -ForegroundColor Red
    exit 1
}
