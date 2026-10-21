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

Desde o serviço comum de execução, cada `latent_dim_N/` do `compare-latent-dims`
também contém `figures/training/`, `figures/reconstructions/` e
`figures/generated_samples/`, além de `figures/latent_space/`.

## Protocolo Fatorial

Raiz: `artifacts/<dataset>/unsupervised/factorial_protocol/run_YYYYMMDD_HHMMSS_<id>/`.
Cada execução cria uma raiz nova; nada é sobrescrito.

```
run_YYYYMMDD_HHMMSS_<id>/
├── shared/
│   ├── environment.json
│   ├── class_distribution.csv
│   ├── projection_sample_indices.csv
│   ├── reconstruction_sample_indices.csv
│   └── generation_seed.json
├── grid/
│   ├── small/    latent_dim_2/  latent_dim_4/  latent_dim_8/  latent_dim_16/  latent_dim_32/
│   ├── medium/   latent_dim_2/  latent_dim_4/  latent_dim_8/  latent_dim_16/  latent_dim_32/
│   └── deep/     latent_dim_2/  latent_dim_4/  latent_dim_8/  latent_dim_16/  latent_dim_32/
├── selection/
│   ├── factorial_comparison.csv
│   ├── selected_configuration.yaml
│   └── selection_summary.md
├── final_evaluation/
│   ├── figures/{reconstructions,generated_samples,latent_space}/
│   ├── latent_data/
│   ├── tables/{final_test_metrics.csv, projection_metrics.csv, pca_explained_variance.csv}
│   ├── final_evaluation_results.json
│   ├── run_summary.md
│   └── artifact_manifest.json
├── supervised_extension/
│   ├── training/
│   ├── final_evaluation/
│   ├── comparison_with_unsupervised.csv
│   ├── comparison_summary.md
│   └── artifact_manifest.json
├── generated_configs/supervised_selected_configuration.yaml
├── logs/protocol.log
├── protocol_config_resolved.yaml
├── protocol_state.json
├── protocol_manifest.json
├── protocol_summary.md
└── scientific_evidence_index.md
```

Cada `grid/<topologia>/latent_dim_<N>/` contém `checkpoints/`
(`best_model.pt`, `checkpoint_metadata.json`), `figures/` (`training/`,
`reconstructions/`, `generated_samples/`, `latent_space/`), `latent_data/`,
`tables/` (`training_history.csv/.json`, `projection_metrics.csv`,
`pca_explained_variance.csv` para N > 2), `config_resolved.yaml`, `metrics.json`,
`run_summary.md` e `artifact_manifest.json`. Um diretório de combinação
interrompido é preservado como `<nome>_incomplete_<timestamp>` na retomada.

| Figura | Arquivos (PNG e SVG) |
|--------|----------------------|
| Treinamento | `training/training_total_loss`, `training/training_loss_components` |
| Reconstruções | `reconstructions/reconstruction_comparison` |
| Amostras geradas | `generated_samples/generated_samples` |
| Espaço latente 2D | `latent_space/latent_space_direct_2d` |
| Espaço latente N > 2 (4, 8, 16, 32) | `latent_space/latent_space_{pca,umap,tsne}_Nd`, `latent_space/pca_explained_variance_Nd` |

A grade padrão tem 15 combinações por dataset (3 topologias × 2, 4, 8, 16 e 32); o
número de diretórios `latent_dim_<N>/` acompanha `factorial_comparison.latent_dimensions`.
Para a dimensão 4 os arquivos são `latent_space_pca_4d`, `latent_space_umap_4d`,
`latent_space_tsne_4d` e `pca_explained_variance_4d` (PNG e SVG), além de
`tables/pca_explained_variance.csv`. A 4D **não** tem gráfico direto.

`selection/factorial_comparison.csv` tem uma linha por combinação, com dataset,
tipo de experimento, topologia, `hidden_dimensions`, dimensão, seed,
`parameter_count`, melhor época, losses de validação, durações, caminhos
(checkpoint, diretório, figuras, projeção, métricas, resumo, manifesto), variância
explicada do PCA, status e contagem de warnings/errors. Campos não aplicáveis
ficam vazios.

`protocol_manifest.json` referencia os manifestos locais (não duplica cada
arquivo interno) e registra protocolo, dataset, identificador, configuração,
ambiente, arquivos compartilhados, combinações, seleção, avaliação final,
extensão supervisionada, relatórios, status, tamanhos, hashes, timestamps,
warnings e errors.

`scientific_evidence_index.md` organiza as evidências por categoria e traz o
questionário para o pesquisador, sem observações visuais preenchidas.
Artefatos de testes e smoke tests (diretório `smoke_test`/`synthetic`,
`data_origin = synthetic`) trazem o aviso "SYNTHETIC DATA: smoke test,
non-scientific results" nos resumos e nunca são resultados científicos.
