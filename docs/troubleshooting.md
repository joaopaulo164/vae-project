# Solução de Problemas

## ModuleNotFoundError: No module named 'vae_project'

Execute o projeto em editable mode:
\\\powershell
pip install -e '.[dev]'
\\\

## CUDA requested but not available

Defina no YAML: \untime.device: cpu\ ou \untime.device: auto\

## Execution Policy (PowerShell)

\\\powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
\\\

## num_workers error no Windows

Mantenha \
um_workers: 0\ (padrão). Aumentar pode causar problemas no Windows.

## Download falhou / sem conexão

Na primeira execução é necessária conexão para baixar os datasets.
Após o download, os dados ficam em data/ e podem ser reutilizados.

## Out of Memory (OOM) com UMAP/t-SNE

Reduza max_samples em projections.sampling.max_samples.
UMAP com low_memory=True já é ativado automaticamente.

## ArtifactError: Run directory already exists

Cada run usa timestamp único. Se ocorrer, aguarde 1 segundo e tente novamente.

## Testes falhando

\\\powershell
python -m pytest -v --tb=long
\\\
Verifique se o ambiente virtual está ativo e as dependências instaladas.
