# Avaliação

Este documento descreve como o espaço latente e o modelo são avaliados, o que cada
saída significa e o que ela **não** permite concluir. Para a leitura dos resultados,
veja [`results_interpretation.md`](results_interpretation.md). Convenção de idioma:
esta documentação é em português; código, logs e os resumos gerados
(`run_summary.md`, `latent_dimension_comparison_summary.md`) são em inglês.

## 1. O que cada fluxo avalia

**`train` / `run-pipeline`** (`ExperimentRunner`): avalia no split de **validação**,
depois de recarregar o melhor checkpoint:

1. figuras de reconstrução e de amostras geradas;
2. extração latente (`latent_data/`);
3. projeções (gráfico direto 2D, ou PCA/UMAP/t-SNE);
4. métricas de agrupamento baseadas em rótulos no espaço latente original
   (`tables/projection_metrics.csv`).

**`compare-latent-dims`**: para **cada** dimensão (2, 8, 16, 32) treina um VAE novo,
seleciona o melhor checkpoint por loss total de validação, **recarrega** esse
checkpoint em um modelo novo e só então extrai `latent_mean` e gera as evidências.
A análise nunca usa os pesos da última época. Os dois fluxos usam o mesmo serviço
(`experiments/latent_evidence.py`).

O classificador linear, as métricas numéricas de reconstrução, ARI/NMI do K-Means e
a trustworthiness estão implementados, mas **não** fazem parte desses fluxos:
`VAEEvaluator` e `run_final_evaluation` precisam ser chamados a partir de Python.

## 2. Extração latente

[`extract_latent_representations`](../src/vae_project/evaluation/latent_extraction.py)
coloca o modelo em `eval()`, percorre o loader em ordem e grava, para cada imagem:

- `latent_mean` (codificação determinística; usada em todas as análises abaixo);
- `latent_log_variance`;
- `latent_sample`: amostra explícita `mu + eps * exp(0.5 * log_var)`. Essas amostras
  são aleatórias e **não** têm seed próprio na função; mudam entre chamadas se o
  estado global do RNG for diferente.

Saída: `latent_data/latent_representations.csv`, com as colunas `sample_index`,
`split`, `label`, `class_name`, `latent_mean_0..L-1`, `latent_log_var_0..L-1`,
`latent_sample_0..L-1`, mais `seed`, `experiment_type` e `latent_dimension` na
comparação; e `.npz` se `artifacts.save_latent_npz` for verdadeiro. `sample_index`
é a posição na ordem do loader do split, não o índice original do dataset.

Os rótulos servem para colorir e calcular métricas; **não** são entradas do modelo
não supervisionado.

## 3. Projeções

Todas as projeções usam `latent_mean` de uma subamostra estratificada de, no máximo,
`projections.sampling.max_samples` (padrão 5.000) pontos de validação. As posições
escolhidas ficam em `latent_data/projection_sample_indices.csv`. Como a seleção
depende só dos rótulos e do seed, **todas as dimensões recebem os mesmos índices**.
A subamostra é balanceada por classe, então as frequências desenhadas podem diferir
das reais; os títulos das figuras indicam quando uma amostra foi usada.

| Dimensão latente | Evidência visual |
|---|---|
| 2 | Gráfico **direto** de `latent_mean_0` × `latent_mean_1`. Não se executa PCA, UMAP nem t-SNE como substituto, e não há variância explicada. |
| 8, 16, 32 | PCA, UMAP e t-SNE (PNG e SVG, com a dimensão no nome) mais o gráfico de variância explicada **do PCA**. |

### PCA e variância explicada

O PCA é linear. `tables/pca_explained_variance.csv` guarda, para as duas primeiras
componentes, `explained_variance_ratio` e o valor acumulado, além de
`latent_dimension`, dataset, seed e tipo de experimento. É a fração da variância
linear **total** das médias latentes capturada; não é uma medida de separação de
classes. Como só duas componentes são ajustadas, o valor não mostra como a variância
se distribui nas demais. **Somente o PCA possui variância explicada.**

### Limitações de UMAP e t-SNE

- Nenhum dos dois fornece variância explicada; nunca a reporte para eles.
- Preservam vizinhanças **locais**; layout global, distâncias entre grupos, tamanhos e
  densidades não são confiáveis.
- Dependem de hiperparâmetros (`n_neighbors`, `min_dist`; `perplexity`, `init`,
  iterações) e de seed.
- Podem mostrar agrupamentos aparentes que não são separáveis no espaço original, e
  fragmentar distribuições contínuas.
- Custo e memória do t-SNE crescem rapidamente com N; o UMAP exige `umap-learn`.
- Mitigação: reportar parâmetros e n, comparar com o PCA e variar parâmetros.

## 4. Métricas de agrupamento (subsidiárias)

[`compute_clustering_metrics`](../src/vae_project/evaluation/clustering_metrics.py)
usa os **rótulos verdadeiros como atribuição de clusters**:

| Coluna em `projection_metrics.csv` | Significado |
|---|---|
| `space_type` | `latent_original`, `pca_2d`, `umap_2d` ou `tsne_2d`. |
| `scope` | `full_split` ou `sample`: sobre todo o split ou sobre a amostra. |
| `silhouette_score` | De −1 a 1; maior = grupos rotulados mais compactos e separados. |
| `calinski_harabasz_index` | Maior = maior dispersão entre grupos do que dentro deles. |
| `davies_bouldin_index` | Menor = grupos mais bem separados. |
| `trustworthiness` | Preservação de vizinhanças na projeção; vazia onde não se aplica (espaço original). |
| `unavailable_metrics` | Métricas não aplicáveis ou que falharam. |

Valores não aplicáveis ficam **vazios**, nunca zero. As métricas descrevem a
concordância entre rótulos e geometria; não provam a existência de clusters naturais,
são sensíveis à dimensão latente e **nunca substituem a análise visual**.

## 5. Classificador linear (sonda)

`train_linear_classifier` ajusta uma regressão logística multinomial sobre as médias
latentes **padronizadas** do treino (o scaler é ajustado só no treino) e informa
acurácia de treino e validação (e de teste apenas se arrays de teste forem
fornecidos). Mede **separabilidade linear**, que não é necessária nem suficiente
para haver agrupamentos. É chamado pelo `VAEEvaluator` quando
`metrics.compute_linear_classifier` é verdadeiro.

## 6. Qualidade de reconstrução

- **Losses de treino/validação**: termos de reconstrução e KL por época em
  `training_history.csv`; dependem do tipo de loss, de `reduction` e de `beta`.
- **Visual**: `reconstruction_comparison` (8 exemplos, a partir da média posterior) e
  `generated_samples` (decodificação de amostras do prior).
- **Numérica**: `compute_reconstruction_metrics` devolve `mean_mse` e `mean_bce` por
  pixel, na escala por elemento (não a soma usada no treino).

Boa reconstrução não implica um espaço latente informativo para as classes, e vice-versa.

## 7. Checkpoint e avaliação

A seleção do melhor checkpoint usa **apenas** a loss total de validação. Para as
evidências, o checkpoint é carregado no modo **inferência**
(`load_for_inference`/`load_best`), que exige pesos e metadados e valida dimensão
latente e topologia; checkpoints antigos (só pesos) continuam aceitos. A retomada de
treino é outro cenário e está descrita em [`reproducibility.md`](reproducibility.md).

## 8. Validação x teste

A avaliação em validação é segura durante a seleção de modelo. A avaliação de teste
(`run_final_evaluation`) é reservada ao final e marcada `evaluation_type: final_test`.
Use a mesma configuração que gerou o checkpoint; caso contrário, o carregamento falha
ou compara a arquitetura errada.

## 9. Dados sintéticos

Testes e smoke test usam dados sintéticos para verificar o **funcionamento** do
código (arquivos criados, formatos, registro no manifesto). Os testes de figuras
conferem existência, tamanho, extensão, dimensão associada, dados de entrada e
registro no manifesto; eles **não** avaliam a qualidade científica. Os artefatos
sintéticos são identificados (`smoke_test`/`synthetic`, `data_origin = synthetic`) e
não entram em relatórios científicos.

## 10. Checklist

- Em qual split cada número foi calculado?
- Houve subamostragem (`scope`, `n`)?
- Qual representação latente (média ou amostra)?
- Quais parâmetros de projeção?
- A métrica é baseada em rótulos?
- Alguma decisão usou dados de teste?
- Os dados são reais (MNIST/Fashion-MNIST) ou sintéticos?
