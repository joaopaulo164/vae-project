# Catálogo de Evidências

## Figuras Geradas

| Arquivo | Conteúdo | Uso Sugerido |
|---------|---------|-------------|
| training_total_loss.png | Curvas de loss treino/validação | Metodologia / Resultados |
| training_loss_components.png | Reconstruction + KL separados | Metodologia / Resultados |
| reconstruction_comparison.png | Originais vs reconstruções | Resultados |
| generated_samples.png | Amostras geradas do prior | Resultados |
| latent_space_direct_2d.png | Espaço latente 2D direto | Resultados (latent_dim=2) |
| latent_space_pca_Nd.png | Projeção PCA colorida | Resultados |
| latent_space_umap_Nd.png | Projeção UMAP colorida | Resultados |
| latent_space_tsne_Nd.png | Projeção t-SNE colorida | Resultados |
| pca_explained_variance_Nd.png | Variância explicada por componente | Resultados |
| latent_traversal_2d.png | Manifold traversal (latent_dim=2) | Resultados (opcional) |

## Tabelas Geradas

| Arquivo | Conteúdo |
|---------|---------|
| training_history.csv | Loss por época (treino + validação) |
| class_distribution.csv | Amostras por classe e por split |
| pca_explained_variance.csv | Variância explicada por componente PCA |
| projection_metrics.csv | Métricas subsidiárias de clustering |
| topology_selection_results.csv | Comparação de topologias |
| latent_dimension_comparison.csv | Comparação de dimensões latentes |

## Metadados por Execução

| Arquivo | Conteúdo |
|---------|---------|
| config_resolved.yaml | Configuração completa resolvida |
| environment.json | Versões, hardware, git hash |
| run_summary.md | Sumário legível da execução |
| artifact_manifest.json | Catálogo de todos os arquivos gerados |
| checkpoints/best_model.pt | Pesos do melhor modelo |
| checkpoints/checkpoint_metadata.json | Metadados do melhor checkpoint |
