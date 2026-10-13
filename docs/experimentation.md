# Protocolo Experimental

## Fase A: Seleção de Topologia

Treina `small`, `medium` e `deep` com a mesma dimensão latente, seed, taxa de
aprendizado etc. Seleciona a topologia com a menor `val_total_loss`.

Comando: `select-topology`

## Fase B: Comparação de Dimensões Latentes

Pressupõe que a topologia já foi selecionada na Fase A. O comando **não** repete
a seleção entre `small`, `medium` e `deep`; ele usa a topologia indicada em
`latent_dimension_comparison.fixed_topology` (padrão `medium`, com
`hidden_dimensions: [512, 256]`). Se `fixed_topology` não coincidir com
`model.topology.name`, o comando aborta com uma mensagem explicativa.

Comando: `compare-latent-dims`

```powershell
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_mnist.yaml
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_fashion_mnist.yaml
```

### Configuração

```yaml
model:
  latent_dimension: 16          # um único modelo: sempre um inteiro

latent_dimension_comparison:
  enabled: true
  fixed_topology: medium
  dimensions: [2, 8, 16, 32]    # cada item é um treinamento independente
```

A lista de dimensões existe somente no bloco `latent_dimension_comparison`.
São rejeitadas: lista vazia, valores menores que 2, duplicatas, valores não
inteiros e `model.latent_dimension` como lista.

### Independência entre dimensões

Para cada dimensão o comando:

1. aplica novamente a configuração de reprodutibilidade (seed e modo determinístico);
2. cria um VAE novo (nenhum modelo é reaproveitado);
3. recria otimizador, scheduler e early stopping (dentro de `VAETrainer`);
4. treina e valida;
5. seleciona o melhor checkpoint pela `val_total_loss`;
6. carrega esse checkpoint em um novo modelo da mesma dimensão (o checkpoint é
   rejeitado se a dimensão gravada nos metadados for diferente);
7. extrai o `latent_mean` do conjunto de validação;
8. gera figuras, tabelas e métricas subsidiárias;
9. grava `config_resolved.yaml`, `metrics.json`, `run_summary.md` e
   `artifact_manifest.json`.

Ao final, os resultados de todas as dimensões são consolidados.

A **única variável experimental é a dimensão latente**. A divisão treino/validação
(`splits.seed: 42`), o seed global, os hiperparâmetros, a topologia, os parâmetros
de UMAP e t-SNE e a estratégia de amostragem são idênticos entre as dimensões.

### Visualizações por dimensão

| Dimensão | Evidência visual |
|----------|------------------|
| 2 | Gráfico direto de `latent_mean_0` × `latent_mean_1`. Não se usa PCA, UMAP ou t-SNE como substituto. |
| 8, 16, 32 | PCA, UMAP, t-SNE e variância explicada do PCA, em PNG e SVG, com a dimensão no nome do arquivo. |

Somente o PCA possui variância explicada (`tables/pca_explained_variance.csv`).
UMAP e t-SNE não a possuem.

### Amostragem das projeções

As projeções usam uma amostra estratificada, configurável e com seed fixo
(`projections.sampling`). Os índices ficam em
`latent_data/projection_sample_indices.csv`. Como a seleção depende apenas dos
rótulos e do seed, todas as dimensões recebem os mesmos índices. As figuras
indicam no título quando uma amostra foi usada.

### Tabelas e métricas subsidiárias

`tables/projection_metrics.csv` distingue o espaço latente original, PCA, UMAP e
t-SNE (`space_type`), indica se a métrica foi calculada sobre a amostra ou sobre
todo o split (`scope`) e traz dimensão, dataset, split, seed e tipo de
experimento. Métricas não aplicáveis ou que falharam ficam **vazias** (nunca
zero) e são listadas em `unavailable_metrics`.

As métricas são subsidiárias: a análise visual é a evidência principal.

### Tratamento de falhas

| Situação | Status | Comportamento |
|----------|--------|---------------|
| Falha no treinamento ou no checkpoint | `failed` | Artefatos parciais são mantidos; segue para a próxima dimensão; a dimensão não entra na seleção. |
| Falha em uma projeção, figura ou métrica | `partial` | Checkpoint e representações latentes são mantidos; o erro vai para o manifesto e o resumo. |
| Nenhuma falha | `completed` | — |

Se todas as dimensões falharem, o comando termina com erro. Cada erro registra
operação, dimensão, dataset, causa provável e o caminho do log
(`latent_dim_N/logs/experiment.log`).

### Resultados consolidados

- `latent_dimension_comparison.csv`: uma linha por dimensão.
- `latent_dimension_comparison_summary.md`: resumo em texto. Indica a dimensão com
  menor `val_total_loss` entre as dimensões que não falharam, mas **não** conclui
  qual dimensão forma melhores agrupamentos; isso exige análise visual.

Valores não aplicáveis (por exemplo, caminho do PCA para a dimensão 2) ficam
vazios na tabela.

### Diretório de cada execução e proteção contra sobrescrita

Cada execução de `compare-latent-dims` grava em um diretório exclusivo,
`.../latent_dimension_comparison/run_YYYYMMDD_HHMMSS_<id>/`, contendo
`latent_dim_N/`, `latent_dimension_comparison.csv` e
`latent_dimension_comparison_summary.md`. `--output-dir` define o diretório
*base*; um `run_...` existente é rejeitado com mensagem de conflito, salvo com
`--overwrite` explícito. Execuções anteriores permanecem intactas. Detalhes em
[`execution.md`](execution.md).

### Dados sintéticos

Dados sintéticos só aparecem em testes e no smoke test. Eles são identificados no
diretório (`smoke_test`/`synthetic`), nos resumos (aviso de que não são resultados
científicos) e na coluna `data_origin`. A comparação científica usa apenas MNIST e
Fashion-MNIST, e métricas sintéticas nunca entram em relatórios científicos.

## Fase C: Avaliação Final

Carrega o melhor checkpoint e avalia **uma vez** no conjunto de teste. Resultados
claramente marcados como métricas de teste.

Módulo: `experiments/final_evaluation.py`

## Experimento Supervisionado (opcional)

Habilitado por `supervised_component.enabled: true`, em uma configuração
independente, executada manualmente depois da escolha da dimensão latente.
Os resultados ficam em diretório separado (`supervised/`) e nunca devem ser
misturados com os do experimento não supervisionado.

Os arquivos `configs/latent_dimensions_*.yaml` mantêm
`supervised_component.enabled: false`. Nesse caso os rótulos não entram na função
de custo nem na atualização dos pesos, e a classification loss é registrada como
zero. Os artefatos são identificados como `unsupervised`.
