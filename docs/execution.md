# Execução

Todos os comandos abaixo são para PowerShell, com o ambiente virtual ativado
(veja [`installation.md`](installation.md)).

## Classificação dos comandos

| Etapa | Comando | Papel |
|-------|---------|-------|
| 1. Execução individual | `train`, `run-pipeline` | Serviço técnico: uma topologia, uma dimensão, um modo |
| 2. Seleção especializada de topologias | `select-topology` | Compara topologias para uma dimensão fixa |
| 3. Comparação especializada de dimensões | `compare-latent-dims` | Compara dimensões para uma topologia fixa |
| 4. **Protocolo fatorial principal** | `run-factorial-protocol` | 3 topologias × 5 dimensões, seleção, teste da vencedora e extensão supervisionada |
| 5. Avaliação final | fase do protocolo | Teste somente da vencedora |
| 6. Extensão supervisionada | fase do protocolo | Mesmo dataset, topologia, dimensão e seed, modelo novo |

Todos usam o mesmo serviço Python (`execute_single_experiment`); nenhum comando
chama outro por subprocesso.

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

# Comparar dimensões latentes (treino, reconstruções, amostras, espaço latente, métricas e tabelas por dimensão)
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_mnist.yaml
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_fashion_mnist.yaml

# Protocolo fatorial principal (MNIST e Fashion-MNIST)
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_fashion_mnist.yaml

# Smoke test (dados sintéticos, sem downloads)
python -m vae_project.cli smoke-test
```

## Protocolo fatorial principal

```powershell
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml
```

| Opção | Efeito |
|---|---|
| `--config` | Configuração do protocolo (obrigatória) |
| `--output-dir BASE` | Diretório **base**; a rodada cria `BASE/run_<timestamp>_<id>/`. Nunca habilita sobrescrita |
| `--run-name NOME` | Usa `NOME` (sanitizado) como identificador no lugar do sufixo aleatório |
| `--resume RUN_DIR` | Retoma a rodada existente em `RUN_DIR` |

Fases executadas, nesta ordem:

1. validação da configuração e criação do diretório exclusivo da rodada;
2. ambiente e arquivos compartilhados (`shared/`);
3. as combinações não supervisionadas da grade (15 na configuração padrão:
   3 topologias × 2, 4, 8, 16 e 32; o total acompanha a configuração),
   em `grid/<topologia>/latent_dim_<N>/`;
4. tabela fatorial e seleção da vencedora (`selection/`), somente com a validação;
5. avaliação da vencedora no teste, uma única vez (`final_evaluation/`);
6. configuração supervisionada (`generated_configs/`) e, no modo `automatic`,
   treinamento e avaliação no teste da extensão (`supervised_extension/`);
7. comparação entre os modos, `protocol_summary.md`,
   `scientific_evidence_index.md`, `protocol_manifest.json` e estado final.

Código de saída: `1` apenas quando o status final é `failed` (por exemplo, nenhuma
combinação válida ou falha da avaliação final); caso contrário, `0`. Os status
possíveis são `completed`, `completed_with_warnings`, `failed`, `interrupted` e
`awaiting_manual_supervised_execution`.

### Tempo de execução e CUDA

Em CPU a rodada é longa: 15 treinamentos completos (até 50 épocas cada, com early
stopping), PCA, UMAP e t-SNE para cada dimensão maior que 2 (inclusive 4), mais o
treinamento supervisionado. Os dois datasets somam 30 treinamentos de grade. Com GPU
NVIDIA, `runtime.device: auto` utiliza CUDA; a execução em CPU continua sendo
suportada.

### Retomada

Se a rodada for interrompida (por exemplo, `Ctrl+C`) ou falhar, use o diretório
`run_...` impresso no início:

```powershell
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml --resume artifacts/mnist/unsupervised/factorial_protocol/run_YYYYMMDD_HHMMSS_identificador
```

Na retomada o protocolo: lê `protocol_state.json`; valida o hash da configuração e
o dataset (configuração diferente é rejeitada); confere cada combinação marcada
como concluída (status, checkpoint, `metrics.json`, `run_summary.md`, manifesto e
evidências mínimas); ignora as íntegras; reinicia as incompletas (o diretório
parcial é preservado como `<nome>_incomplete_<timestamp>`); não sobrescreve
arquivos válidos; e registra a retomada em `resumed_at`. Uma rodada com status
`completed` não pode ser retomada.

Uma rodada iniciada com a grade histórica de 12 combinações (`[2, 8, 16, 32]`) **não
pode ser retomada** com a configuração padrão de 15 combinações: o hash da
configuração difere e a retomada é recusada, para que os resultados das duas grades
não se misturem. Inicie uma nova rodada; as anteriores permanecem intactas.

Uma rodada que falhou e foi retomada com sucesso termina como
`completed_with_warnings`, porque os erros anteriores permanecem registrados.

### Modo manual da extensão supervisionada

Com `supervised_extension.execution_mode: manual` (e `enabled: false`), o
protocolo não treina o modelo supervisionado. Ele grava
`generated_configs/supervised_selected_configuration.yaml`, informa o comando de
execução manual (`run-pipeline` com essa configuração) e termina com status
`awaiting_manual_supervised_execution`.

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
- `run_summary.md`, `latent_dimension_comparison_summary.md`,
  `selection_summary.md`, `protocol_summary.md`, `comparison_summary.md` e
  `scientific_evidence_index.md`: inglês (são artefatos gerados pelo código).

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

Protocolo fatorial:

```
artifacts/{dataset}/unsupervised/factorial_protocol/
└── run_{YYYYMMDD_HHMMSS}_{id}/
    ├── shared/
    ├── grid/{small|medium|deep}/latent_dim_{2|4|8|16|32}/
    ├── selection/
    ├── final_evaluation/
    ├── supervised_extension/
    ├── generated_configs/
    ├── protocol_config_resolved.yaml
    ├── protocol_state.json
    ├── protocol_manifest.json
    ├── protocol_summary.md
    └── scientific_evidence_index.md
```

O catálogo completo de arquivos está em [`evidence_catalog.md`](evidence_catalog.md).
