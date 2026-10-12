# Guia de Evidências para Artigo Científico

Este documento mapeia perguntas de pesquisa a artefatos gerados pelo projeto.
**Não preencha as colunas de resultados antes de executar os experimentos.**

## Mapeamento: Pergunta ? Evidência ? Arquivo

| Pergunta de Pesquisa | Evidência Necessária | Arquivo Gerado | Figura/Tabela | Cuidados |
|----------------------|---------------------|----------------|---------------|---------|
| Há formação visual de clusters? | Projeções coloridas por classe | figures/latent_space/ | latent_space_pca/umap/tsne_Nd.png | Verificar em múltiplas projeções |
| Quais classes se sobrepõem? | Análise visual das projeções | figures/latent_space/ | Todas as projeções | Artefato de projeção vs. sobreposição real |
| Quanto de variância o PCA explica? | pca_explained_variance.csv | tables/ | pca_explained_variance_Nd.png | Válido apenas para PCA |
| Qual topologia tem menor val loss? | topology_selection_results.csv | topology_selection/ | Tabela | Um seed; variabilidade não estimada |
| Dimensão latente afeta separação? | latent_dimension_comparison.csv | dimension_comparison/ | Tabela | Capacidade cresce com dimensão |
| MNIST vs Fashion-MNIST diferem? | Comparar figuras dos dois datasets | artifacts/mnist/ vs artifacts/fashion_mnist/ | Projeções side-by-side | Rodar mesma config nos dois |
| Componente supervisionado ajuda? | Comparar unsupervised vs supervised | artifacts/unsupervised/ vs supervised/ | Figuras lado a lado | Artefatos em diretórios separados |

## Evidências para Seção de Metodologia

- config_resolved.yaml: parâmetros exatos do experimento
- environment.json: versões de bibliotecas e hardware
- tables/class_distribution.csv: distribuição de classes por split

## Evidências para Seção de Resultados

- figures/training/: curvas de treinamento
- figures/reconstructions/: qualidade de reconstrução
- figures/latent_space/: análise do espaço latente
- tables/projection_metrics.csv: métricas subsidiárias

## Limitações a Declarar no Artigo

1. Um seed por configuração — variabilidade não estimada
2. Arquitetura MLP totalmente conectada — sem estrutura espacial
3. Métricas de clustering usam rótulos como clusters, não clusters descobertos
4. t-SNE e UMAP: não reportar variância explicada
5. Resultados podem diferir entre versões de PyTorch/CUDA/hardware
