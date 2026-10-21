# Estrutura do Projeto

## src/vae_project/

| Módulo | Responsabilidade |
|--------|----------------|
| cli.py | Interface de linha de comando (argparse) |
| config/ | Carregamento, validação e modelos de configuração |
| data/ | Datasets, divisões, loaders, transforms |
| models/ | VAE, componentes MLP, cabeça supervisionada |
| training/ | Trainer, losses, early stopping, checkpointing, histórico |
| evaluation/ | Extração latente, métricas de clustering, classificação, reconstrução |
| visualization/ | Plots de treino, reconstrução, espaço latente, configuração visual |
| projections/ | PCA, UMAP, t-SNE, subsampling |
| experiments/ | Serviço de execução individual, seleção de topologia, comparação de dimensões, protocolo fatorial, evidências latentes, avaliação no teste |
| reporting/ | Manifesto de artefatos, sumário de run, relatório comparativo, relatórios do protocolo, comparação supervisionado x não supervisionado |
| utils/ | Device, paths, logging, timing, reprodutibilidade, ambiente |

### Módulos de experimentos

| Arquivo | Responsabilidade |
|---------|------------------|
| experiments/single_experiment.py | **Serviço comum** `execute_single_experiment`: valida, treina, recarrega o melhor checkpoint e gera todas as evidências de uma configuração |
| experiments/experiment_result.py | `SingleExperimentResult` (resultado tipado), `EvidenceProfile` e status |
| experiments/model_evidence.py | Reconstruções e amostras geradas a partir do melhor checkpoint |
| experiments/latent_evidence.py | Extração do `latent_mean`, figuras do espaço latente, projeções, tabelas e métricas subsidiárias |
| experiments/runner.py | `train` e `run-pipeline` (usa o serviço comum) |
| experiments/topology_selection.py | `select-topology`: seleção entre `small`, `medium` e `deep` (usa o serviço comum) |
| experiments/latent_dimension_comparison.py | `compare-latent-dims`: um VAE independente por dimensão (usa o serviço comum) |
| experiments/factorial_protocol.py | **Orquestrador** do `run-factorial-protocol` |
| experiments/protocol_models.py | Grade 3×4, `FactorialCombination`, `CombinationRecord`, validação de artefatos |
| experiments/protocol_selection.py | Tabela fatorial, seleção da vencedora, `selected_configuration.yaml`, `selection_summary.md` |
| experiments/protocol_state.py | `protocol_state.json`, hash da configuração, validação da retomada |
| experiments/protocol_shared.py | Arquivos compartilhados (`shared/`): índices e seed de geração |
| experiments/heldout_evaluation.py | Avaliação no teste da vencedora e da extensão supervisionada |
| experiments/final_evaluation.py | Avaliação avulsa no conjunto de teste (uso via Python) |
| reporting/protocol_report.py | `protocol_summary.md`, `scientific_evidence_index.md`, `protocol_manifest.json` |
| reporting/supervised_comparison.py | `comparison_with_unsupervised.csv`, `comparison_summary.md` |
| reporting/experiment_summary.py | `run_summary.md` de cada execução |

## configs/

Arquivos YAML para cada experimento. Separados por dataset, tipo de experimento
e objetivo (seleção de topologia, comparação de dimensões, etc.). O protocolo
principal usa `factorial_protocol_mnist.yaml` e
`factorial_protocol_fashion_mnist.yaml`, cuja grade padrão é 3 topologias × 5
dimensões (2, 4, 8, 16 e 32) = 15 configurações por dataset. Os arquivos
`latent_dimensions_*.yaml` (comparação especializada) continuam com `[2, 8, 16, 32]`.

## tests/

- unit/: testes rápidos sem I/O de dados reais
- integration/: testes com dados sintéticos, loop completo
- conftest.py: fixtures compartilhadas (dados sintéticos)

## artifacts/

Gerado em runtime. Nunca versionado. Cada run em subdiretório único; a comparação
de dimensões usa `latent_dimension_comparison/run_<timestamp>_<id>/latent_dim_N/`
e nunca reutiliza um `run_...` existente sem `--overwrite`. O protocolo fatorial
usa `<dataset>/unsupervised/factorial_protocol/run_<timestamp>_<id>/` (estrutura em
[`evidence_catalog.md`](evidence_catalog.md)) e não possui opção de sobrescrita.

## data/

Datasets baixados em runtime. Nunca versionado.
