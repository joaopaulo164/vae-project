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

### model
- latent_dimension: dimensão do espaço latente
- topology.name e topology.hidden_dimensions

### loss
- reconstruction_loss_type: bce (padrão) | mse
- beta: peso do KL (padrão 1.0 = VAE padrão)
- reduction: sum | mean

### supervised_component
- enabled: false (padrão) | true
- classification_loss_weight
- hidden_dimensions
- use_latent_mean

### training
- max_epochs, batch_size, shuffle_train

### early_stopping
- enabled, patience, min_delta

### runtime
- device: auto | cpu | cuda
- cuda_fallback_to_cpu

### reproducibility
- global_seed, deterministic_mode

### projections
- methods: [pca, umap, tsne]
- umap.n_neighbors, umap.min_dist
- tsne.perplexity, tsne.n_iter
- sampling.max_samples

Veja configs/base.yaml para todos os valores padrão documentados.
