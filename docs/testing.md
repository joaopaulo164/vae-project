# Testes Automatizados

## Marcadores

- \unit\: testes unitários rápidos (sem downloads, sem GPU)
- \integration\: testes de integração com dados sintéticos
- \slow\: testes lentos (treinamento completo)
- \equires_cuda\: requer GPU NVIDIA

## Executar Testes

\\\powershell
# Todos os testes (unit + integration)
python -m pytest

# Apenas unitários
python -m pytest -m unit

# Com cobertura
python -m pytest --cov=vae_project --cov-report=term-missing
\\\

## Arquivos de Teste

| Arquivo | O que testa |
|---------|-------------|
| unit/test_config.py | Carregamento e validação de YAML |
| unit/test_vae.py | Arquitetura do VAE, shapes |
| unit/test_losses.py | BCE, MSE, KL divergence, total loss |
| unit/test_reparameterization.py | Trick de reparametrização |
| unit/test_early_stopping.py | Parada antecipada |
| unit/test_model_checkpointing.py | Salvar/carregar checkpoint |
| unit/test_projections.py | PCA, t-SNE, subsampling |
| unit/test_data_splits.py | Divisão reprodutível dos dados |
| unit/test_metrics.py | Métricas de clustering subsidiárias |
| unit/test_checkpointing.py | Utilitários de paths |
| integration/test_training_pipeline.py | Loop de treinamento completo |
| integration/test_evaluation_pipeline.py | Extração latente, PCA, manifesto |
| integration/test_smoke_cli.py | CLI smoke test e validate-config |
