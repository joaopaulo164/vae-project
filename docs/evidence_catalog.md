# Catálogo de Evidências

## Figuras Geradas

| Arquivo | Conteúdo | Uso Sugerido |
|---------|---------|-------------|
| training_total_loss.png | Curvas de loss treino/validação | Metodologia / Resultados |
| training_loss_components.png | Reconstruction + KL separados | Metodologia / Resultados |
| reconstruction_comparison.png | Originais vs reconstruções | Resultados |
| generated_samples.png | Amostras geradas do prior | Resultados |
| latent_space_direct_2d.png | Espaço latente 2D direto (`latent_mean_0` × `latent_mean_1`) | Resultados (latent_dim=2) |
| latent_space_pca_Nd.png | Projeção PCA colorida | Resultados |
| latent_space_umap_Nd.png | Projeção UMAP colorida | Resultados |
| latent_space_tsne_Nd.png | Projeção t-SNE colorida | Resultados |
| pca_explained_variance_Nd.png | Variância explicada por componente (somente PCA) | Resultados |
| latent_traversal_2d.png | Manifold traversal (latent_dim=2) | Resultados (opcional) |

Cada figura é salva em PNG e SVG. Em `Nd`, `N` é a dimensão latente (por exemplo,
`latent_space_pca_16d.png`). Os títulos das projeções indicam o dataset, o split, o
tipo de experimento (`unsupervised`/`supervised`) e, quando houve amostragem, o
número de pontos usados ("sample of N").

## Tabelas Geradas

| Arquivo | Conteúdo |
|---------|---------|
| training_history.csv | Loss por época (treino + validação) |
| class_distribution.csv | Amostras por classe e por split |
| pca_explained_variance.csv | Variância explicada por componente PCA (`latent_dimension`, `component`, `explained_variance_ratio`, `cumulative_explained_variance_ratio`, `n_samples` e contexto) |
| projection_metrics.csv | Métricas subsidiárias por espaço (`space_type`: original, PCA, UMAP, t-SNE) e escopo (`scope`: amostra ou split completo) |
| projection_sample_indices.csv | Índices da amostra estratificada usada nas projeções (em `latent_data/`) |
| topology_selection_results.csv | Comparação de topologias |
| latent_dimension_comparison.csv | Consolidação das dimensões latentes (uma linha por dimensão) |
| latent_dimension_comparison_summary.md | Resumo textual da comparação, com status e erros por dimensão |

Métricas e caminhos não aplicáveis ficam vazios, nunca zero. Em
`projection_metrics.csv`, a coluna `unavailable_metrics` lista o que não pôde ser
calculado.

## Metadados por Execução

| Arquivo | Conteúdo |
|---------|---------|
| config_resolved.yaml | Configuração completa resolvida |
| environment.json | Versões, hardware, git hash |
| run_summary.md | Sumário legível da execução |
| metrics.json | Métricas da execução (comparação de dimensões) |
| artifact_manifest.json | Catálogo dos arquivos gerados, com status, data de criação, contexto e hash SHA-256 |
| checkpoints/best_model.pt | Estado completo do melhor modelo: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict` (se houver scheduler), `scaler_state_dict` (se houver AMP), `early_stopping_state`, `epoch`, `best_epoch`, losses de validação, dimensão latente, topologia, seed, dataset, tipo de experimento e `resolved_config` |
| checkpoints/checkpoint_metadata.json | Metadados do melhor checkpoint (época, losses, dimensão latente, seed, tipo de experimento) |

Checkpoints antigos contêm só os pesos (um `state_dict` puro) e continuam
carregáveis para inferência/evidências; veja [`reproducibility.md`](reproducibility.md).

## Comparação de Dimensões Latentes

Estrutura em
`artifacts/<dataset>/<unsupervised|supervised>/latent_dimension_comparison/`.
Cada execução tem seu próprio diretório `run_YYYYMMDD_HHMMSS_<id>/`, que nunca é
reutilizado sem `--overwrite`:

```
run_YYYYMMDD_HHMMSS_<id>/
├── latent_dimension_comparison.csv
├── latent_dimension_comparison_summary.md
└── latent_dim_N/
    ├── checkpoints/
    ├── figures/latent_space/
    ├── latent_data/        (latent_representations.csv/.npz, projection_sample_indices.csv)
    ├── tables/             (training_history, pca_explained_variance, projection_metrics)
    ├── logs/experiment.log
    ├── config_resolved.yaml
    ├── metrics.json
    ├── run_summary.md
    └── artifact_manifest.json
```

Execuções com dados sintéticos (testes e smoke test) usam diretórios com
`smoke_test` ou `synthetic` no caminho, e a coluna `data_origin` do CSV (também
presente em `metrics.json` e no manifesto) vale `synthetic`; esses artefatos
não são resultados científicos. Em execuções reais, `data_origin = real`.

Para `latent_dim_2` não há `pca_explained_variance.csv` nem
`projection_sample_indices.csv`, pois o espaço latente é desenhado diretamente.
