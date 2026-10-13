# Estrutura do Projeto

## src/vae_project/

| Módulo | Responsabilidade |
|--------|----------------|
| cli.py | Interface de linha de comando (argparse) |
| config/ | Carregamento, validação e modelos de configuração |
| data/ | Datasets, divisões, loaders, transforms |
| models/ | VAE, componentes MLP, cabeça supervisionada |
| training/ | Trainer, losses, early stopping, checkpointing, histórico |
| evaluation/ | Extração latente, métricas de clustering, classificação, reconstrução |
| visualization/ | Plots de treino, reconstrução, espaço latente, configuração visual |
| projections/ | PCA, UMAP, t-SNE, subsampling |
| experiments/ | Runner, seleção de topologia, comparação de dimensões, geração de evidências latentes, avaliação final |
| reporting/ | Manifesto de artefatos, sumário de run, relatório comparativo |
| utils/ | Device, paths, logging, timing, reprodutibilidade, ambiente |

### Módulos de experimentos

| Arquivo | Responsabilidade |
|---------|------------------|
| experiments/runner.py | Pipeline de um único treinamento (`train`, `run-pipeline`) |
| experiments/topology_selection.py | Seleção entre `small`, `medium` e `deep` |
| experiments/latent_dimension_comparison.py | Treina um VAE independente por dimensão e consolida os resultados |
| experiments/latent_evidence.py | Serviço compartilhado: extração do `latent_mean`, figuras, projeções, tabelas e métricas subsidiárias de um modelo treinado. Usado pelo `runner.py` e pela comparação de dimensões |
| experiments/final_evaluation.py | Avaliação única no conjunto de teste |

## configs/

Arquivos YAML para cada experimento. Separados por dataset, tipo de experimento
e objetivo (seleção de topologia, comparação de dimensões, etc.)

## tests/

- unit/: testes rápidos sem I/O de dados reais
- integration/: testes com dados sintéticos, loop completo
- conftest.py: fixtures compartilhadas (dados sintéticos)

## artifacts/

Gerado em runtime. Nunca versionado. Cada run em subdiretório único; a comparação
de dimensões usa `latent_dimension_comparison/run_<timestamp>_<id>/latent_dim_N/`
e nunca reutiliza um `run_...` existente sem `--overwrite`.

## data/

Datasets baixados em runtime. Nunca versionado.
