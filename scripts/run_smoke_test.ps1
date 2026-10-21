# run_smoke_test.ps1
# Runs the CLI smoke test (no downloads required, uses synthetic data).
# Run from the project root directory with the virtual environment active.

Write-Host "=== VAE Project: Smoke Test ===" -ForegroundColor Cyan
Write-Host "NOTE: This test uses synthetic data — no internet access required." -ForegroundColor Yellow
Write-Host ""

python -m vae_project.cli smoke-test

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Smoke test PASSED." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Smoke test FAILED. Check the output above for details." -ForegroundColor Red
    exit 1
}
