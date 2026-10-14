# Arquitetura

## Variational Autoencoder (VAE)

O modelo principal é um VAE totalmente conectado implementado em
[`src/vae_project/models/vae.py`](../src/vae_project/models/vae.py).

## Componentes Obrigatórios

| Componente | Variável | Módulo |
|-----------|---------|--------|
| Encoder MLP | `encoder` | `models/components.py` |
| Camada de média latente | `latent_mean_layer` | `models/vae.py` |
| Camada de log-variância | `latent_log_variance_layer` | `models/vae.py` |
| Truque de reparametrização | `reparameterize()` | `models/vae.py` |
| Amostra latente | `latent_sample` | `models/vae.py` |
| Decoder MLP | `decoder` | `models/components.py` |
| Reconstruction loss | `compute_reconstruction_loss()` | `training/losses.py` |
| KL divergence loss | `compute_kl_divergence_loss()` | `training/losses.py` |
| Total VAE loss | `compute_vae_loss()` | `training/losses.py` |

## Topologias Configuráveis

| Nome | hidden_dimensions |
|------|-------------------|
| small | [256, 128] |
| medium | [512, 256] |
| deep | [512, 256, 128] |

O decoder é construído de forma simétrica (`hidden_dimensions` invertido). A
contagem de parâmetros depende da topologia e da dimensão latente e é registrada
em cada execução (`parameter_count`).

## Saída do Forward Pass

O método `forward()` retorna um `VAEOutput` com:

- `reconstruction`: imagem reconstruída em [0, 1];
- `latent_mean`: média da posterior q(z|x);
- `latent_log_variance`: log da variância da posterior;
- `latent_sample`: amostra z via reparametrização.

## Cabeça Supervisionada (opcional)

Implementada em `models/supervised_head.py`. Ativada por
`supervised_component.enabled: true`. Recebe `latent_mean` como entrada e produz
logits de classificação. Na grade fatorial ela fica desativada; só a extensão
supervisionada a utiliza.

## Organização dos pipelines

O projeto separa o **serviço de execução de uma configuração** dos
**orquestradores**:

```
                      run-factorial-protocol (protocolo principal)
                                  │
         ┌────────────────────────┼─────────────────────────┐
         │                        │                         │
  select-topology          compare-latent-dims          run-pipeline / train
 (topologias, dim fixa)   (dimensões, topologia fixa)   (uma configuração)
         └────────────────────────┴─────────────────────────┘
                                  │
                 execute_single_experiment  (serviço comum)
```

| Camada | Módulo | Responsabilidade |
|--------|--------|------------------|
| CLI | `cli.py` | Interpreta argumentos e chama serviços Python; nunca usa subprocesso |
| Serviço comum | `experiments/single_experiment.py` | Treina, recarrega o melhor checkpoint e gera todas as evidências de **uma** configuração; retorna `SingleExperimentResult` |
| Resultado tipado | `experiments/experiment_result.py` | `SingleExperimentResult`, `EvidenceProfile`, status |
| Evidências do modelo | `experiments/model_evidence.py`, `experiments/latent_evidence.py` | Reconstruções, amostras geradas, extração latente, PCA/UMAP/t-SNE, tabelas, métricas |
| Execução individual | `experiments/runner.py` | `train` e `run-pipeline` |
| Seleção de topologia | `experiments/topology_selection.py` | `select-topology` (dimensão fixa) |
| Comparação de dimensões | `experiments/latent_dimension_comparison.py` | `compare-latent-dims` (topologia fixa) |
| Grade e seleção | `experiments/protocol_models.py`, `experiments/protocol_selection.py` | Grade 3×4, tabela fatorial, escolha da vencedora (somente validação) |
| Estado e retomada | `experiments/protocol_state.py` | `protocol_state.json`, hash da configuração, validação da retomada |
| Amostras compartilhadas | `experiments/protocol_shared.py` | Índices de projeção e de reconstrução, seed de geração, ambiente |
| Avaliação no teste | `experiments/heldout_evaluation.py` | Avaliação única da vencedora e da extensão supervisionada |
| Orquestrador | `experiments/factorial_protocol.py` | Une todas as fases do protocolo |
| Relatórios | `reporting/protocol_report.py`, `reporting/supervised_comparison.py`, `reporting/experiment_summary.py`, `reporting/artifact_manifest.py` | Resumos, índice de evidências, manifestos, comparação entre modos |

Regras de projeto:

- o orquestrador do protocolo **não** invoca comandos de CLI por subprocesso;
- PCA, UMAP e t-SNE existem uma única vez (`projections/`) e são reutilizados;
- cada combinação da grade cria modelo, otimizador, scheduler e early stopping
  novos e reaplica o seed;
- toda evidência posterior ao treino usa o **melhor checkpoint** (menor
  `val_total_loss`).

Veja [`experimentation.md`](experimentation.md) para o protocolo e
[`project_structure.md`](project_structure.md) para os módulos.
