# Guia de Evidências para Artigo Científico

Este documento mapeia perguntas de pesquisa aos artefatos gerados pelo projeto.
**Não preencha resultados antes de executar os experimentos.** Convenção de idioma:
esta documentação é em português; os resumos gerados (`run_summary.md`,
`latent_dimension_comparison_summary.md`) são em inglês.

Todos os caminhos de comparação são relativos a
`artifacts/<dataset>/unsupervised/latent_dimension_comparison/run_<timestamp>_<id>/`
(abreviado como `RUN/`). Cada execução tem seu próprio `RUN/`; cite no artigo o nome
exato do diretório usado.

## Regra de elegibilidade

Somente execuções com `data_origin = real` (MNIST e Fashion-MNIST) podem sustentar
afirmações científicas. Artefatos de testes e do smoke test (`data_origin =
synthetic`, diretório com `smoke_test`/`synthetic`) **não** devem ser citados.

## Mapeamento: pergunta → evidência → arquivo

| Pergunta de pesquisa | Evidência | Arquivo | Cuidados |
|---|---|---|---|
| Há formação visual de clusters? | Projeções coloridas por classe | `RUN/latent_dim_N/figures/latent_space/` (`latent_space_direct_2d`, `latent_space_pca_Nd`, `latent_space_umap_Nd`, `latent_space_tsne_Nd`) | Verificar em mais de uma projeção; para N=2 usar o gráfico direto |
| Quais classes se sobrepõem? | Análise visual das projeções | mesmas figuras | Distinguir artefato de projeção de sobreposição real |
| Quanto da variância o PCA explica? | `tables/pca_explained_variance.csv` e `pca_explained_variance_Nd` | `RUN/latent_dim_N/` (N > 2) | Válido **apenas** para PCA; não existe para N=2, UMAP ou t-SNE |
| Qual dimensão tem menor loss de validação? | `latent_dimension_comparison.csv` | `RUN/` | Afirmação numérica sobre a loss; não diz qual dimensão agrupa melhor. Um seed |
| A dimensão latente afeta a separação? | Figuras de todas as dimensões lado a lado + `tables/projection_metrics.csv` | `RUN/latent_dim_*/` | Métricas são subsidiárias; capacidade cresce com a dimensão |
| MNIST e Fashion-MNIST diferem? | Comparar os `RUN/` dos dois datasets | `artifacts/mnist/...` e `artifacts/fashion_mnist/...` | Mesma configuração nos dois |
| O componente supervisionado ajuda? | Comparar execuções não supervisionada e supervisionada | diretórios `unsupervised/` e `supervised/` | Experimento independente, executado manualmente; nunca misturar |
| Qual topologia tem menor val loss? | `topology_selection_results.csv` | `topology_selection/` | Um seed; variabilidade não estimada |

## Evidências para a seção de Metodologia

- `latent_dim_N/config_resolved.yaml`: parâmetros exatos de cada dimensão;
- `latent_dim_N/checkpoints/checkpoint_metadata.json` e `best_model.pt`
  (`resolved_config`, seed, topologia, dimensão latente);
- `latent_dim_N/artifact_manifest.json`: arquivos, tamanhos e SHA-256;
- `latent_dim_N/latent_data/projection_sample_indices.csv`: amostra idêntica entre
  dimensões;
- `environment.json` (execuções `train`/`run-pipeline`): versões e hardware.

## Evidências para a seção de Resultados

- `latent_dim_N/tables/training_history.csv`: curvas de treinamento;
- `latent_dim_N/figures/latent_space/`: espaço latente (PNG e SVG);
- `latent_dim_N/tables/projection_metrics.csv`: métricas subsidiárias;
- `latent_dim_N/run_summary.md` e `latent_dimension_comparison_summary.md`:
  resumos (em inglês);
- `latent_dimension_comparison.csv`: tabela consolidada.

## Limitações a declarar

1. Um seed por configuração: a variabilidade entre seeds não é estimada.
2. Arquitetura MLP totalmente conectada, sem estrutura espacial.
3. As métricas usam os rótulos como clusters, não clusters descobertos.
4. UMAP e t-SNE distorcem distâncias globais; não reportar variância explicada
   para eles.
5. A comparação principal é não supervisionada; o componente supervisionado é um
   experimento separado.
6. Os resultados podem variar entre versões de PyTorch/CUDA e hardware.
7. A seleção do modelo usa apenas o split de validação; o teste não é usado.
