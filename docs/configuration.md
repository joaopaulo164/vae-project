# Configuração

## Estrutura dos Arquivos YAML

Todos os arquivos YAML em configs/ herdam os valores padrão implicitamente.
Apenas os valores que diferem do padrão precisam ser especificados.

## Seções Principais

### project
- name, description, version, quick_mode

### dataset
- name: mnist | fashion_mnist
- download: true/false
- num_workers: 0 (padrão seguro Windows)

### splits
- seed: 42 (divisão treino/validação; fixa entre dimensões na comparação)
- validation_fraction, stratified

### model
- latent_dimension: dimensão do espaço latente de **um único modelo** (inteiro)
- topology.name e topology.hidden_dimensions

### loss
- reconstruction_loss_type: bce (padrão) | mse
- beta: peso do KL (padrão 1.0 = VAE padrão)
- reduction: sum | mean

`beta != 1.0` **não** é erro de configuração (β-VAE é válido); apenas `beta <= 0`
é rejeitado. O desvio do VAE padrão é registrado como aviso (`logger.warning`) pelo
`VAETrainer` ao iniciar o treino, e não pela validação.

### supervised_component
- enabled: false (padrão) | true
- classification_loss_weight
- hidden_dimensions
- use_latent_mean

Com `enabled: false`, os rótulos não participam da loss nem da atualização dos
pesos, e a classification loss é registrada como zero. Os artefatos são
identificados como `unsupervised`.

### training
- max_epochs, batch_size, shuffle_train

### early_stopping
- enabled, patience, min_delta

### runtime
- device: auto | cpu | cuda
- cuda_fallback_to_cpu

`device: cuda` é sempre aceito pela validação do YAML; a disponibilidade de CUDA só
pode ser verificada em tempo de execução e é tratada por `get_device`, que levanta
`DeviceSetupError` com instruções (ou cai para CPU se `cuda_fallback_to_cpu: true`).

### reproducibility
- global_seed: 42 (um único seed; não há lista de seeds)
- deterministic_mode: true nos arquivos de comparação de dimensões

### projections
- methods: [pca, umap, tsne]
- umap.n_neighbors, umap.min_dist, umap.random_state (42)
- tsne.perplexity, tsne.max_iter, tsne.random_state (42)
- sampling.enabled, sampling.max_samples (5000), sampling.stratified, sampling.seed (42)

A amostragem é estratificada e depende apenas dos rótulos e do seed. Por isso todas
as dimensões latentes da comparação recebem os mesmos índices, gravados em
`latent_data/projection_sample_indices.csv`.

### latent_dimension_comparison

```yaml
latent_dimension_comparison:
  enabled: true
  fixed_topology: medium
  dimensions: [2, 8, 16, 32]
```

- `dimensions`: lista de dimensões; cada item é um treinamento independente. É o
  único local onde uma lista de dimensões é permitida.
- `fixed_topology`: nome da topologia já selecionada. Deve coincidir com
  `model.topology.name`. A comparação não repete a seleção de topologia.
- `enabled`: se `false`, o comando termina com erro explicativo.

Validação: lista vazia, valores menores que 2, duplicatas e valores não inteiros
são rejeitados, assim como `model.latent_dimension` com uma lista.

### Protocolo fatorial

`configs/factorial_protocol_mnist.yaml` e
`configs/factorial_protocol_fashion_mnist.yaml` usam o mesmo carregador e a
mesma validação dos demais arquivos (herdam os padrões e só declaram as
diferenças). Os blocos novos são:

```yaml
protocol:
  name: mnist_vae_factorial_protocol
  resume_if_interrupted: true               # permite --resume
  prevent_overwrite: true                   # cada rodada cria um run_... exclusivo
  continue_on_noncritical_evidence_error: true

factorial_comparison:
  enabled: true
  topologies: [small, medium, deep]
  latent_dimensions: [2, 8, 16, 32]
  generate_full_evidence: true
  selection:
    primary_metric: validation_total_loss
    direction: minimize
    tolerance: 1.0e-6
    tie_breakers:
      - metric: validation_reconstruction_loss
        direction: minimize
      - metric: parameter_count
        direction: minimize
      - metric: latent_dimension
        direction: minimize
      - metric: topology_name
        direction: ascending

final_evaluation:
  enabled: true
  split: test                               # qualquer outro valor é rejeitado
  use_selected_configuration: true
  use_best_checkpoint: true
  retrain_selected_configuration: false

supervised_extension:
  enabled: true
  execution_mode: automatic                 # ou manual (com enabled: false)
  use_selected_configuration: true
  classification_loss_weight: 1.0
  use_latent_mean: true
```

Os campos `model.latent_dimension` e `model.topology` do arquivo do protocolo são
apenas valores iniciais: cada combinação da grade os substitui. As topologias
`small`, `medium` e `deep` têm as `hidden_dimensions` descritas em
[`architecture.md`](architecture.md). A configuração mantém
`supervised_component.enabled: false` para a grade; a extensão supervisionada
deriva sua própria configuração a partir da vencedora.

Validações do protocolo (a configuração é rejeitada com mensagem explicativa):

- lista de topologias vazia, topologia desconhecida ou duplicada;
- lista de dimensões vazia, duplicada, menor que 2 ou com valores não inteiros;
- `primary_metric` desconhecida, `direction` inválida, `tolerance` negativa;
- critério de desempate desconhecido ou com direção inválida;
- `supervised_component.enabled: true` na grade (a grade é não supervisionada);
- `final_evaluation.split` diferente de `test` e
  `retrain_selected_configuration: true` (não suportado);
- `execution_mode` desconhecido, ou extensão supervisionada sem classificação
  configurada (peso da classification loss não positivo ou
  `supervised_component.hidden_dimensions` vazio);
- combinações duplicadas (topologias repetidas).

O hash da configuração (`protocol_state.json`) detecta mudanças entre a execução
original e a retomada.

Veja configs/base.yaml para todos os valores padrão documentados.
