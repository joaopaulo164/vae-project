# Protocolo Experimental

O **protocolo científico principal** é o `run-factorial-protocol`, descrito na
próxima seção. As Fases A e B (`select-topology` e `compare-latent-dims`) continuam
disponíveis como análises especializadas e isoladas; o protocolo fatorial **não**
as usa como fases obrigatórias nem as invoca por subprocesso.

## Protocolo Fatorial Principal

```powershell
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml

python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_fashion_mnist.yaml
```

### Grade

3 topologias × 5 dimensões latentes = **15 configurações por dataset** e
**30 treinamentos de grade** nos dois datasets:

| Topologia | hidden_dimensions | Dimensões latentes |
|-----------|-------------------|--------------------|
| small | [256, 128] | 2, 4, 8, 16, 32 |
| medium | [512, 256] | 2, 4, 8, 16, 32 |
| deep | [512, 256, 128] | 2, 4, 8, 16, 32 |

A grade é gerada dinamicamente a partir de `factorial_comparison.topologies` e
`factorial_comparison.latent_dimensions`; configurações históricas que declaram
apenas `[2, 8, 16, 32]` continuam executando 12 combinações.

**Por que a dimensão 4:** ela preenche o intervalo entre 2D e 8D, cria a progressão
2, 4, 8, 16 e 32 e permite avaliar os ganhos marginais com o aumento da dimensão.
Não há resultado presumido antes dos experimentos: se 4D é suficiente, melhor ou pior
que 2D e 8D é uma pergunta empírica. A 4D é tratada como qualquer dimensão maior que
2 (sem visualização direta; PCA, UMAP e t-SNE), e suas três combinações (small, medium
e deep) concorrem à seleção pelo mesmo critério de validação.

Somente a topologia, suas `hidden_dimensions` e a dimensão latente variam.
Permanecem constantes: dataset, divisão dos dados, seed, batch size, learning
rate, otimizador, scheduler, épocas máximas, patience, min delta, reconstruction
loss, beta, normalização, critério do checkpoint e do early stopping, parâmetros
de PCA/UMAP/t-SNE, estratégia e quantidade de amostras das projeções e política
de dispositivo. Há **um único seed** (`reproducibility.global_seed: 42`,
`splits.seed: 42`, `random_state: 42` em UMAP e t-SNE); não há lista de seeds.

Cada combinação cria modelo, otimizador, scheduler e early stopping novos,
reaplica o seed, treina de forma independente e grava seu próprio checkpoint.
Nenhum peso é reaproveitado entre combinações.

### Fase 1: grade não supervisionada

Todas as combinações da grade (15 na configuração padrão) usam
`supervised_component.enabled: false`: os rótulos
não participam da loss nem dos gradientes e a classification loss é ausente/zero.
Os rótulos só são usados depois do treino para colorir figuras, nomear classes e
calcular métricas subsidiárias. Os artefatos são identificados como
`unsupervised`.

Cada combinação gera, a partir do **melhor checkpoint** (menor
`val_total_loss`, recarregado após o treino):

| Categoria | Arquivos |
|-----------|----------|
| `figures/training/` | `training_total_loss`, `training_loss_components` (PNG e SVG) |
| `figures/reconstructions/` | `reconstruction_comparison` (PNG e SVG) |
| `figures/generated_samples/` | `generated_samples` (PNG e SVG) |
| `figures/latent_space/` | 2D: `latent_space_direct_2d`; 4/8/16/32: `latent_space_pca_Nd`, `latent_space_umap_Nd`, `latent_space_tsne_Nd`, `pca_explained_variance_Nd` |

Para a dimensão 2 o espaço latente é desenhado diretamente
(`latent_mean_0` × `latent_mean_1`); PCA, UMAP e t-SNE nunca o substituem.
Somente o PCA possui variância explicada (`tables/pca_explained_variance.csv`).

Os mesmos exemplos de projeção, os mesmos índices de reconstrução e o mesmo seed
de geração são usados em todas as combinações do dataset (`shared/`). Os valores
numéricos latentes mudam entre os modelos; os exemplos de origem, não.

### Fase 2: seleção da vencedora

`selection/factorial_comparison.csv` contém exatamente uma linha por combinação.
A vencedora é escolhida **exclusivamente com a validação**:

1. menor `validation_total_loss` (tolerância configurável, padrão `1.0e-6`);
2. desempate: menor `validation_reconstruction_loss`;
3. menor `parameter_count`;
4. menor `latent_dimension`;
5. ordenação estável pelo nome da topologia.

São excluídas execuções com falha, incompletas, sem checkpoint ou com valores não
finitos. Se nenhuma for válida, o protocolo termina como `failed` **antes** do
teste e da extensão supervisionada. A aparência visual de UMAP/t-SNE e as métricas
do teste nunca participam da seleção. Saídas: `selected_configuration.yaml` e
`selection_summary.md`.

### Fase 3: avaliação final no teste

Somente a vencedora é avaliada no teste, uma única vez, com o melhor checkpoint da
fase de grade (**sem retreinar**). As outras 11 combinações nunca são avaliadas no
teste. O teste não altera topologia, dimensão, hiperparâmetros, melhor época,
checkpoint nem a decisão de seleção. Os resultados vão para `final_evaluation/`
(`tables/final_test_metrics.csv`, `final_evaluation_results.json`, figuras,
`run_summary.md`, `artifact_manifest.json`), marcados como `split: test`,
`phase: final evaluation` e `selected configuration: true`. Se a avaliação falhar,
a seleção é preservada, nenhuma outra combinação é escolhida e a avaliação pode
ser retomada com `--resume`.

### Fase 4: extensão supervisionada

Com `supervised_extension.enabled: true` e `execution_mode: automatic`, o
protocolo treina um modelo **novo** com a mesma topologia, `hidden_dimensions`,
dimensão latente, seed, dataset, divisões, batch size, learning rate, otimizador,
scheduler, épocas, patience, reconstruction loss, beta, normalização e parâmetros
das projeções. Só mudam: `supervised_component.enabled: true`, a cabeça
classificadora, a classification loss e seu peso, o tipo do experimento e o
diretório. Os pesos do VAE não supervisionado **não** são usados como
inicialização. A loss é

```
total supervised loss = reconstruction loss + beta × KL divergence
                        + classification_loss_weight × classification loss
```

O melhor checkpoint supervisionado é avaliado no teste e comparado com o
não supervisionado em `supervised_extension/comparison_with_unsupervised.csv` e
`comparison_summary.md`. A `total_loss` supervisionada **não** é comparada
numericamente com a não supervisionada, pois a composição é diferente; comparam-se
separadamente reconstruction loss, KL, reconstruções, organização visual,
métricas subsidiárias e desempenho classificatório (acurácia, precision, recall,
F1 e matriz de confusão). Campos de interpretação manual ficam em branco para o
pesquisador.

Com `enabled: false` e `execution_mode: manual`, nada é treinado: o protocolo
grava `generated_configs/supervised_selected_configuration.yaml`, informa o
comando manual e termina com `awaiting_manual_supervised_execution`. A
configuração gerada é gravada nos dois modos, para rastreabilidade.

### Fase 5: relatórios

`protocol_summary.md`, `scientific_evidence_index.md` (com perguntas para o
pesquisador: existem clusters? os rótulos estão separados? quais classes se
sobrepõem? qual topologia reconstrói melhor? como a dimensão afeta a organização?
a menor validation loss coincide com a melhor separação visual? a supervisão
melhora a separação? a supervisão afeta a reconstrução?),
`protocol_manifest.json` e `protocol_state.json`. Nenhuma observação visual é
preenchida automaticamente.

### Falhas

| Situação | Comportamento |
|----------|---------------|
| Falha no treino de uma combinação | Marcada `failed`, artefatos parciais preservados, fora da seleção, segue para a próxima |
| Falha em projeção/figura/métrica após o treino | `partial` (equivalente a `completed_with_warnings`); checkpoint, histórico e representações preservados; continua elegível se a loss de validação for válida |
| Nenhuma combinação válida | Protocolo `failed`; sem teste e sem extensão supervisionada |
| Falha na avaliação final | Seleção preservada; nenhuma outra combinação é escolhida; retomável |

### Estado e retomada

`protocol_state.json` guarda identificador, hash da configuração, dataset, status,
fase atual, combinações concluídas/falhas/pendentes/atual, seleção, estados da
avaliação final e da extensão supervisionada, timestamps, warnings e errors.
`--resume <run_dir>` valida hash, dataset e artefatos reais antes de reaproveitar
qualquer combinação (veja [`execution.md`](execution.md)).

## Fase A: Seleção de Topologia (análise especializada)

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
independente (`run-pipeline`). Dentro do protocolo fatorial, é a extensão
supervisionada da Fase 4.
Os resultados ficam em diretório separado (`supervised/`) e nunca devem ser
misturados com os do experimento não supervisionado.

Os arquivos `configs/latent_dimensions_*.yaml` mantêm
`supervised_component.enabled: false`. Nesse caso os rótulos não entram na função
de custo nem na atualização dos pesos, e a classification loss é registrada como
zero. Os artefatos são identificados como `unsupervised`.
