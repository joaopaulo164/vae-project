# Solução de Problemas

## ModuleNotFoundError: No module named 'vae_project'

Instale o projeto em modo editável:

```powershell
pip install -e ".[dev]"
```

## CUDA requested but not available

Defina no YAML `runtime.device: cpu` ou `runtime.device: auto`.

## Execution Policy (PowerShell)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## num_workers error no Windows

Mantenha `dataset.num_workers: 0` (padrão). Aumentar pode causar problemas no
Windows.

## Download falhou / sem conexão

Na primeira execução é necessária conexão para baixar os datasets. Após o
download, os dados ficam em `data/` e podem ser reutilizados.

## Out of Memory (OOM) com UMAP/t-SNE

Reduza `projections.sampling.max_samples`. O UMAP com `low_memory=True` já é
ativado automaticamente.

## ArtifactError: Run directory already exists

Cada execução usa um diretório exclusivo (`run_<timestamp>_<id>`). Se o erro
ocorrer, confira se `--output-dir` aponta para o diretório **base** e não para um
`run_...` existente. No `compare-latent-dims`, somente `--overwrite` permite
reutilizar um `run_...`; o `run-factorial-protocol` não possui essa opção.

## Protocolo fatorial: duração elevada

O `run-factorial-protocol` treina 12 modelos, mais um supervisionado, por dataset.
Em CPU isso pode levar horas. Use GPU NVIDIA (`runtime.device: auto`) se houver e
reduza `projections.sampling.max_samples` para acelerar UMAP e t-SNE. Não altere as
variáveis controladas (épocas, batch size etc.) se os datasets forem comparados.

## Protocolo fatorial: rodada interrompida ou com falha

Execute o mesmo comando com `--resume`, apontando para o diretório `run_...` da
rodada (impresso no início da execução e presente em
`artifacts/<dataset>/unsupervised/factorial_protocol/`):

```powershell
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml --resume artifacts/mnist/unsupervised/factorial_protocol/run_YYYYMMDD_HHMMSS_identificador
```

Mensagens comuns:

| Mensagem | Causa | Ação |
|----------|-------|------|
| `protocol_state.json` ausente ou ilegível | O diretório não é uma rodada do protocolo | Confira o caminho do `run_...` |
| Hash de configuração diferente | A configuração mudou desde a rodada original | Use a configuração original ou inicie uma rodada nova |
| `Cannot resume: the protocol run already finished` | Rodada com status `completed` | Nada a retomar; inicie uma nova rodada se necessário |
| `resume is disabled` | `protocol.resume_if_interrupted: false` | Habilite no YAML |
| Dados reais x sintéticos | A rodada foi criada com outro tipo de dados | Retome com o mesmo tipo de dados |

Combinações incompletas são reiniciadas; o diretório parcial é preservado como
`<nome>_incomplete_<timestamp>` para inspeção. Uma combinação só é considerada
concluída se status, checkpoint, `metrics.json`, `run_summary.md`, manifesto e
evidências mínimas forem válidos; a mera existência do diretório não basta.

## Protocolo fatorial: nenhuma combinação válida

Se todas as combinações falharem, o protocolo termina como `failed` antes do teste
e da extensão supervisionada. Consulte `protocol_summary.md`, `logs/protocol.log` e
o `run_summary.md` de cada combinação (seção de erros).

## Protocolo fatorial: avaliação final falhou

A seleção é preservada e nenhuma outra combinação é escolhida. Corrija a causa
(por exemplo, dataset ausente em `data/`) e use `--resume`.

## Protocolo fatorial: rodada termina como `completed_with_warnings`

Ocorre quando alguma combinação ficou `partial`/`failed` ou quando há warnings ou
erros registrados em `protocol_state.json` (inclusive de uma tentativa anterior
retomada). Veja `protocol_summary.md`.

## Projeção ausente em uma combinação

Uma falha de PCA, UMAP ou t-SNE não invalida o treinamento: a combinação fica
`partial`, mantém checkpoint, histórico e representações latentes e continua
elegível para a seleção. O erro aparece no `run_summary.md` e no manifesto.

## Testes falhando

```powershell
python -m pytest -v --tb=long
```

Verifique se o ambiente virtual está ativo e as dependências instaladas. Os testes
do protocolo fatorial treinam VAEs pequenos com dados sintéticos e levam alguns
minutos em CPU.
