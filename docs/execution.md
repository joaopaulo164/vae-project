# Execução

Todos os comandos abaixo são para PowerShell, com o ambiente virtual ativado
(veja [`installation.md`](installation.md)).

## Comandos CLI

```powershell
# Validar configuração
python -m vae_project.cli validate-config --config configs/mnist_unsupervised.yaml

# Treinar
python -m vae_project.cli train --config configs/mnist_unsupervised.yaml

# Pipeline completo (treino + avaliação + visualização)
python -m vae_project.cli run-pipeline --config configs/mnist_unsupervised.yaml

# Selecionar topologia
python -m vae_project.cli select-topology --config configs/topology_selection_mnist.yaml

# Comparar dimensões latentes (treino, figuras, métricas e tabelas por dimensão)
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_mnist.yaml
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_fashion_mnist.yaml

# Smoke test (dados sintéticos, sem downloads)
python -m vae_project.cli smoke-test
```

## Comparação de dimensões latentes

Cada execução completa de `compare-latent-dims` cria um diretório **exclusivo**
`run_YYYYMMDD_HHMMSS_<identificador>/` dentro do diretório base
`artifacts/<dataset>/<unsupervised|supervised>/latent_dimension_comparison/`.
O identificador é um sufixo hexadecimal aleatório (ou o valor de `--run-name`,
sanitizado). O nome não contém caracteres proibidos no Windows, e duas
execuções iniciadas no mesmo segundo nunca compartilham o diretório.

| Opção | Efeito |
|---|---|
| `--output-dir BASE` | `BASE` é o diretório **base**; a execução cria `BASE/run_<timestamp>_<id>/`. |
| `--output-dir RUN_EXISTENTE` | Rejeitado com mensagem de conflito (código de saída `1`), sem gravar nada. |
| `--output-dir RUN_EXISTENTE --overwrite` | Reutiliza o diretório e substitui arquivos de mesmo nome. Um aviso é registrado no log. |
| `--run-name NOME` | Usa `NOME` (sanitizado) como identificador no lugar do sufixo aleatório. |

Código de saída:

- `0`: todas as dimensões terminaram com status `completed`;
- `1`: erro de configuração, ao menos uma dimensão `partial` ou `failed`, ou falha geral.

Ao final, o terminal mostra uma tabela com `latent_dimension`, `execution_status`,
`best_epoch`, `validation_total_loss` e `training_time`.

### Tempo de execução

Em CPU, a comparação executa quatro treinamentos completos e, para cada dimensão
maior que 2, PCA, UMAP e t-SNE. O tempo total pode ser longo. A primeira execução
do UMAP compila código com numba (cerca de 30 segundos). Se houver GPU NVIDIA com
CUDA, `runtime.device: auto` a utiliza.

### Reexecução

Reexecutar o mesmo comando **não** sobrescreve a execução anterior: cada chamada
gera um novo `run_<timestamp>_<id>/`. Somente `--overwrite`, junto de um
`--output-dir` que aponte para um diretório de execução existente, permite
substituir arquivos. Sem `--overwrite`, a execução existente permanece intacta.

### Retomar treinamento

O `best_model.pt` guarda o estado completo de treino (otimizador, scheduler,
contador de early stopping, configuração resolvida). `VAETrainer.resume_from_checkpoint()`
restaura esse estado e `fit(..., start_epoch=...)` continua do epoch seguinte ao
do checkpoint. Checkpoints antigos (apenas pesos) continuam servindo para
inferência/evidências, mas a retomada falha com a mensagem
`cannot be used to resume training`. Detalhes em
[`reproducibility.md`](reproducibility.md).

### Smoke test sintético

Os testes automatizados e o smoke test usam dados **sintéticos**. Seus
diretórios precisam conter `smoke_test` ou `synthetic` (a função rejeita outros
nomes), e `run_summary.md`, `latent_dimension_comparison_summary.md`,
`metrics.json`, o manifesto e o CSV trazem `data_origin = synthetic` e o aviso
de que **não são resultados científicos**. A comparação científica usa apenas
MNIST e Fashion-MNIST.

## Convenção de idioma

- Documentação (`README.md` e `docs/`): português.
- Código, identificadores, docstrings, mensagens de log e erros técnicos: inglês.
- `run_summary.md` e `latent_dimension_comparison_summary.md`: inglês
  (são artefatos gerados pelo código).

## Artefatos gerados

Execuções de `train` e `run-pipeline`:

```
artifacts/{dataset}/{unsupervised|supervised}/latent_{dim}/run_{timestamp}/
```

Comparação de dimensões latentes:

```
artifacts/{dataset}/{unsupervised|supervised}/latent_dimension_comparison/
└── run_{YYYYMMDD_HHMMSS}_{id}/
    ├── latent_dimension_comparison.csv
    ├── latent_dimension_comparison_summary.md
    └── latent_dim_{N}/
        ├── checkpoints/
        ├── figures/latent_space/
        ├── latent_data/
        ├── tables/
        ├── logs/experiment.log
        ├── config_resolved.yaml
        ├── metrics.json
        ├── run_summary.md
        └── artifact_manifest.json
```

O catálogo completo de arquivos está em [`evidence_catalog.md`](evidence_catalog.md).
