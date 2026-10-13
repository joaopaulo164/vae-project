# Testes Automatizados

Todos os testes usam dados sintéticos: sem downloads e sem GPU.

## Marcadores

- `unit`: testes unitários rápidos (sem downloads, sem GPU)
- `integration`: testes de integração com dados sintéticos
- `slow`: testes lentos (treinamento completo)
- `requires_cuda`: requer GPU NVIDIA

## Executar Testes

```powershell
# Todos os testes (unit + integration)
python -m pytest

# Apenas unitários
python -m pytest -m unit

# Apenas a comparação de dimensões latentes
python -m pytest tests/integration/test_latent_dimension_comparison.py

# Com cobertura
python -m pytest --cov=vae_project --cov-report=term-missing
```

## Arquivos de Teste

| Arquivo | O que testa |
|---------|-------------|
| unit/test_config.py | Carregamento e validação de YAML, lista de dimensões (vazia, < 2, duplicada, não inteira), `model.latent_dimension` como lista, configs de comparação não supervisionadas e com seed único |
| unit/test_vae.py | Arquitetura do VAE, shapes |
| unit/test_losses.py | BCE, MSE, KL divergence, total loss |
| unit/test_reparameterization.py | Truque de reparametrização |
| unit/test_early_stopping.py | Parada antecipada |
| unit/test_model_checkpointing.py | Salvar/carregar checkpoint, metadados (seed, tipo de experimento), rejeição de dimensão latente incompatível |
| unit/test_checkpoint_resume.py | Conteúdo do checkpoint (modelo, otimizador, scheduler, scaler, early stopping, metadados, configuração), carga para inferência, compatibilidade com checkpoint antigo (só pesos), retomada de treino e mensagem clara ao tentar retomar de checkpoint incompleto, validação de dimensão latente e topologia |
| unit/test_projections.py | PCA, t-SNE, subsampling |
| unit/test_data_splits.py | Divisão reprodutível dos dados |
| unit/test_metrics.py | Métricas de clustering subsidiárias |
| unit/test_checkpointing.py | Utilitários de paths |
| integration/test_training_pipeline.py | Loop de treinamento completo e retomada via `resume_from_checkpoint` |
| integration/test_evaluation_pipeline.py | Extração latente, PCA, manifesto |
| integration/test_latent_dimension_comparison.py | Fluxo completo de `compare-latent-dims` (veja abaixo) |
| integration/test_smoke_cli.py | CLI smoke test e validate-config |

## Comparação de dimensões latentes

`integration/test_latent_dimension_comparison.py` treina VAEs pequenos em CPU por
poucas épocas, com divisões sintéticas injetadas (nenhum dataset é baixado). O
UMAP é substituído por um dublê de teste, porque a primeira execução real compila
código com numba e é lenta; PCA e t-SNE rodam de verdade em entradas pequenas. Os
dublês existem apenas nos testes. O arquivo verifica:

- um diretório e um modelo independente por dimensão, com a mesma topologia e o mesmo seed;
- melhor checkpoint com a dimensão latente correta;
- dimensão 2 com gráfico direto e sem PCA/UMAP/t-SNE nem variância explicada;
- dimensões maiores que 2 com PCA, UMAP, t-SNE e variância explicada, em PNG e SVG,
  com a dimensão no nome;
- `pca_explained_variance.csv`, `projection_metrics.csv` e `projection_sample_indices.csv`
  (amostra estratificada e idêntica entre dimensões);
- manifesto, `run_summary.md`, `config_resolved.yaml` e `metrics.json` por dimensão;
- tabela e resumo consolidados, com valores não aplicáveis vazios;
- experimento não supervisionado, com classification loss igual a zero;
- falha de projeção (status `partial`) e falha de treinamento (status `failed`,
  sem interromper as demais dimensões);
- rejeição de lista vazia, dimensão menor que 2, duplicatas, valores não inteiros,
  topologia incompatível e comparação desabilitada;
- validação **mecânica** dos arquivos de figura (gráfico direto 2D, PCA, UMAP, t-SNE e
  variância explicada do PCA): existência, tamanho maior que zero, extensão PNG/SVG
  com assinatura válida, associação à dimensão correta, dados de entrada não vazios,
  registro no manifesto (com tamanho e SHA-256) e referência na tabela consolidada.
  Esses testes **não** avaliam a qualidade científica das figuras;
- prevenção de sobrescrita: duas execuções consecutivas geram diretórios distintos,
  um `run_...` existente é rejeitado (sem alterar seus arquivos) e só `--overwrite`
  o reutiliza, com aviso; nomes de diretório seguros no Windows; mensagem de conflito
  na CLI;
- rotulagem de dados sintéticos: diretório com `smoke_test`/`synthetic`, aviso nos
  resumos, `data_origin` no CSV, em `metrics.json` e no manifesto.

Os testes de configuração (`unit/test_config.py`) também cobrem os dois trechos que
antes eram `pass` em `validation.py` (veja [`configuration.md`](configuration.md)).

## Idioma

Documentação em português; código, docstrings, logs e os resumos gerados
(`run_summary.md`, `latent_dimension_comparison_summary.md`) em inglês.
