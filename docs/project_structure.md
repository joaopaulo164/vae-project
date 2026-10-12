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
| experiments/ | Runner, seleção de topologia, comparação de dimensões, avaliação final |
| reporting/ | Manifesto de artefatos, sumário de run, relatório comparativo |
| utils/ | Device, paths, logging, timing, reprodutibilidade, ambiente |

## configs/

Arquivos YAML para cada experimento. Separados por dataset, tipo de experimento
e objetivo (seleção de topologia, comparação de dimensões, etc.)

## tests/

- unit/: testes rápidos sem I/O de dados reais
- integration/: testes com dados sintéticos, loop completo
- conftest.py: fixtures compartilhadas (dados sintéticos)

## artifacts/

Gerado em runtime. Nunca versionado. Cada run em subdiretório único.

## data/

Datasets baixados em runtime. Nunca versionado.
