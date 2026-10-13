# Reprodutibilidade

Este documento descreve o que o projeto faz para que uma execução possa ser
repetida, auditada e, quando possível, retomada. Convenção de idioma: esta
documentação é em português; código, docstrings, logs e os resumos gerados
(`run_summary.md`, `latent_dimension_comparison_summary.md`) são em inglês.

## Seed único

Todos os experimentos usam **um único seed** (`reproducibility.global_seed: 42`).
Não há lista de seeds e a variabilidade entre seeds **não é estimada**.

`set_global_seed` (`utils/reproducibility.py`) fixa:

- `random.seed(seed)`;
- `numpy.random.seed(seed)`;
- `torch.manual_seed(seed)` e `torch.cuda.manual_seed_all(seed)`.

Na comparação de dimensões latentes, o seed é aplicado **de novo antes de cada
dimensão**, e cada dimensão cria modelo, otimizador, scheduler e early stopping
novos. A divisão treino/validação usa `splits.seed: 42` e é a mesma para todas as
dimensões. A amostragem das projeções (`projections.sampling.seed: 42`) e os
`random_state` de UMAP e t-SNE (42) também são fixos.

## Modo determinístico

`reproducibility.deterministic_mode: true` (usado nos YAMLs de comparação) ativa
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True`,
`cudnn.benchmark = False` e define `CUBLAS_WORKSPACE_CONFIG` quando necessário.
Em GPU, isso pode tornar operações mais lentas ou falhar se não houver kernel
determinístico.

## Metadados registrados por execução

| Arquivo | Conteúdo |
|---|---|
| `config_resolved.yaml` | Configuração completa resolvida (por dimensão, na comparação) |
| `environment.json` | Versões de bibliotecas, sistema, hardware e hash do git (execuções `train`/`run-pipeline`) |
| `metrics.json` | Métricas, seed, split, status e `data_origin` |
| `run_summary.md` | Resumo legível da execução (em inglês) |
| `artifact_manifest.json` | Arquivos gerados, status, tamanho, data e SHA-256 |
| `checkpoints/checkpoint_metadata.json` | Metadados do melhor checkpoint |

## Checkpoint com estado completo

`checkpoints/best_model.pt` é salvo sempre que a loss total de validação melhora.
Ele guarda, além dos pesos, o estado necessário para retomar o treino:

| Chave | Quando existe |
|---|---|
| `model_state_dict` | sempre |
| `optimizer_state_dict` | quando o otimizador é passado (o `VAETrainer` sempre passa) |
| `scheduler_state_dict` | somente se existe scheduler (`scheduler.enabled: true`) |
| `scaler_state_dict` | somente se um `GradScaler` (AMP) é fornecido; o projeto não usa AMP por padrão |
| `early_stopping_state` | melhor valor, melhor epoch, contador de paciência, flag de disparo |
| `supervised_head_state_dict` | somente se há cabeça supervisionada |
| `epoch`, `best_epoch` | epoch do checkpoint |
| `best_val_total_loss`, `val_reconstruction_loss`, `val_kl_divergence_loss`, `val_classification_loss` | losses de validação do melhor epoch |
| `latent_dimension`, `topology_name`, `hidden_dimensions` | identidade do modelo |
| `global_seed`, `dataset_name`, `experiment_type` | proveniência |
| `resolved_config` | configuração resolvida (tipos primitivos) |
| `metadata` | o mesmo conteúdo de `checkpoint_metadata.json` |

O arquivo contém apenas tensores e tipos primitivos, portanto é carregado com
`torch.load(..., weights_only=True)`.

O estado de early stopping e do scheduler é atualizado **antes** do salvamento,
de modo que o checkpoint representa o estado a partir do qual o treino continua
no epoch seguinte.

### Dois cenários de carga

**Inferência / geração de evidências** — `ModelCheckpointer.load_for_inference`
(`load_best` é um alias mantido para os chamadores existentes). Exige apenas os
pesos e os metadados. Valida `latent_dimension` e `hidden_dimensions` (topologia)
contra o modelo e levanta `CheckpointError` em caso de divergência.

**Retomada de treino** — `ModelCheckpointer.load_for_resume` e
`VAETrainer.resume_from_checkpoint()`. Exige também otimizador e, se existirem no
treino, scheduler, scaler, early stopping e cabeça supervisionada. Depois:

```python
state = trainer.resume_from_checkpoint()
trainer.fit(train_loader, val_loader, dataset_name, start_epoch=state.next_epoch)
```

O treino continua em `best_epoch + 1`. Como só o **melhor** checkpoint é salvo,
epochs executados depois do melhor não são recuperados.

### Compatibilidade com checkpoints antigos

Checkpoints anteriores (um `state_dict` puro em `best_model.pt` mais
`checkpoint_metadata.json`) continuam funcionando **para inferência**. Ao tentar
retomar o treino a partir deles, o projeto não falha em silêncio: registra um
erro e levanta `CheckpointError` com a mensagem
`Checkpoint ... cannot be used to resume training: it does not contain the
required state (optimizer_state_dict, ...)`. Para obter um checkpoint retomável é
preciso treinar de novo.

## Diretórios exclusivos e não sobrescrita

Cada execução completa de `compare-latent-dims` grava em
`.../latent_dimension_comparison/run_YYYYMMDD_HHMMSS_<id>/`. Uma execução nova
nunca reutiliza o diretório de outra. Um `run_...` existente é rejeitado, a menos
que `--overwrite` seja passado explicitamente. Veja [`execution.md`](execution.md).

## Dados sintéticos x experimento real

Os testes e o smoke test usam dados **sintéticos** (tensores aleatórios, sem
download). Eles:

- gravam em diretórios cujo caminho contém `smoke_test` ou `synthetic`;
- declaram nos resumos que **não são resultados científicos**;
- marcam `data_origin = synthetic` no CSV consolidado, em `metrics.json` e no
  manifesto.

A comparação científica usa **somente MNIST e Fashion-MNIST** (`data_origin =
real`). Métricas sintéticas nunca devem ser incorporadas a relatórios científicos.

## Escopo do experimento principal

- Comparação principal **não supervisionada** (`supervised_component.enabled:
  false`): os rótulos não entram na loss nem na atualização dos pesos; servem
  apenas para colorir gráficos e calcular métricas subsidiárias.
- O componente supervisionado é um experimento **independente**, executado
  manualmente depois da escolha da dimensão latente, com resultados em diretório
  separado.
- Topologia fixa (`medium`), selecionada previamente; `model.latent_dimension`
  continua sendo um único inteiro.

## Limitações

Reprodutibilidade bit a bit **não é garantida** entre hardware (CPU x GPU),
versões de CUDA, versões de PyTorch ou de outras bibliotecas, mesmo com seeds
idênticos. Registre `environment.json` e `config_resolved.yaml` junto com os
resultados. UMAP e t-SNE são estocásticos; fixar `random_state` reduz, mas não
elimina, diferenças entre plataformas.
