# Projeto VAE — Variational Autoencoders para MNIST e Fashion-MNIST

Um projeto científico completo, reprodutível e executável para estudo de
**Variational Autoencoders (VAE)** com os datasets MNIST e Fashion-MNIST.

> **Leia antes de executar:** Os resultados produzidos pelo projeto devem ser
> interpretados visualmente antes de recorrer a métricas quantitativas.
> Consulte a seção [Análise Visual e Métricas Subsidiárias](#análise-visual-e-métricas-subsidiárias).

> **Protocolo científico principal:** `run-factorial-protocol` compara
> 3 topologias × 4 dimensões latentes (12 configurações por dataset) e
> automatiza seleção, avaliação final e extensão supervisionada. Veja
> [Protocolo Fatorial Principal](#protocolo-fatorial-principal).

---

## Índice

1. [Descrição do Projeto](#descrição-do-projeto)
2. [Objetivos Acadêmicos](#objetivos-acadêmicos)
3. [Datasets Utilizados](#datasets-utilizados)
4. [Arquitetura Geral](#arquitetura-geral)
5. [Estrutura do Repositório](#estrutura-do-repositório)
6. [Pré-requisitos](#pré-requisitos)
7. [Instalação no Windows](#instalação-no-windows)
8. [Execução Rápida (Smoke Test)](#execução-rápida-smoke-test)
9. [Classificação dos Pipelines](#classificação-dos-pipelines)
10. [Execução Individual](#execução-individual)
11. [Seleção de Topologia](#seleção-de-topologia)
12. [Comparação de Dimensões Latentes](#comparação-de-dimensões-latentes)
13. [Protocolo Fatorial Principal](#protocolo-fatorial-principal)
14. [Componente Supervisionado Opcional](#componente-supervisionado-opcional)
15. [Testes Automatizados](#testes-automatizados)
16. [Qualidade de Código](#qualidade-de-código)
17. [Localização dos Artefatos](#localização-dos-artefatos)
18. [Análise Visual e Métricas Subsidiárias](#análise-visual-e-métricas-subsidiárias)
19. [Limitações de PCA, UMAP e t-SNE](#limitações-de-pca-umap-e-t-sne)
20. [Reprodutibilidade](#reprodutibilidade)
21. [Mapa da Documentação](#mapa-da-documentação)
22. [Solução de Problemas](#solução-de-problemas)
23. [Licença e Citação](#licença-e-citação)

---

## Descrição do Projeto

O projeto implementa um **Variational Autoencoder (VAE)** totalmente conectado
para estudo do espaço latente aprendido sobre imagens 28×28 dos datasets
MNIST (dígitos) e Fashion-MNIST (roupas).

O VAE aprende a mapear cada imagem para uma distribuição no espaço latente
(via `latent_mean` e `latent_log_variance`) e a reconstruir a imagem a partir
de uma amostra latente (`latent_sample`). Ao treinar sem utilizar os rótulos,
o projeto investiga se estruturas coerentes emergem espontaneamente no espaço
latente e como essas estruturas podem ser visualizadas e analisadas.

---

## Objetivos Acadêmicos

| Objetivo | Módulo principal | Configuração | Artefatos |
|---|---|---|---|
| Comparar 3 topologias × 4 dimensões latentes (protocolo principal) | [`experiments/factorial_protocol.py`](src/vae_project/experiments/factorial_protocol.py) | [`factorial_protocol_mnist.yaml`](configs/factorial_protocol_mnist.yaml) | `factorial_protocol/run_*/` |
| Treinar VAE com diferentes topologias | [`experiments/topology_selection.py`](src/vae_project/experiments/topology_selection.py) | [`topology_selection_mnist.yaml`](configs/topology_selection_mnist.yaml) | `topology_*/training_history.csv` |
| Selecionar melhor modelo por validation loss | [`training/early_stopping.py`](src/vae_project/training/early_stopping.py) | `early_stopping.patience` | `checkpoints/best_model.pt` |
| Analisar espaço latente 2D diretamente | [`visualization/latent_plots.py`](src/vae_project/visualization/latent_plots.py) | `model.latent_dimension: 2` | `latent_space_direct_2d.png` |
| PCA em dimensões superiores | [`projections/pca.py`](src/vae_project/projections/pca.py) | `projections.methods: [pca]` | `latent_space_pca_Nd.png` |
| UMAP em dimensões superiores | [`projections/umap_projection.py`](src/vae_project/projections/umap_projection.py) | `projections.umap` | `latent_space_umap_Nd.png` |
| t-SNE em dimensões superiores | [`projections/tsne.py`](src/vae_project/projections/tsne.py) | `projections.tsne` | `latent_space_tsne_Nd.png` |
| Comparar MNIST vs Fashion-MNIST | [`experiments/runner.py`](src/vae_project/experiments/runner.py) | configs separados | `artifacts/mnist/` e `artifacts/fashion_mnist/` |
| Extensão supervisionada opcional | [`models/supervised_head.py`](src/vae_project/models/supervised_head.py) | `supervised_component.enabled: true` | Diretório `supervised/` |

---

## Datasets Utilizados

| Dataset | Classes | Tamanho treino | Tamanho teste |
|---|---|---|---|
| MNIST | 10 dígitos (0–9) | 60.000 | 10.000 |
| Fashion-MNIST | 10 categorias de roupas | 60.000 | 10.000 |

Classes do Fashion-MNIST: T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, Ankle boot.

Os datasets são baixados via `torchvision.datasets` na primeira execução e
armazenados em `data/`. O diretório `data/` **não é versionado no Git**.

Consulte [`docs/datasets.md`](docs/datasets.md) para detalhes da divisão
treino/validação/teste e política anti-data leakage.

---

## Arquitetura Geral

```
Input (784) → Encoder MLP → [latent_mean, latent_log_variance]
                           → Reparameterization Trick → latent_sample
                           → Decoder MLP → Reconstruction (784, Sigmoid)
```

**Componentes obrigatórios do VAE:**

- [`encoder`](src/vae_project/models/vae.py): MLP de entrada ao espaço oculto
- [`latent_mean_layer`](src/vae_project/models/vae.py): µ da posterior q(z|x)
- [`latent_log_variance_layer`](src/vae_project/models/vae.py): log σ² da posterior
- [`reparameterize()`](src/vae_project/models/vae.py): z = µ + ε·σ (trick)
- [`decoder`](src/vae_project/models/vae.py): MLP de z para reconstrução
- [`reconstruction_loss`](src/vae_project/training/losses.py): BCE (padrão) ou MSE
- [`kl_divergence_loss`](src/vae_project/training/losses.py): KL(q||p) analítica
- [`total_vae_loss`](src/vae_project/training/losses.py): recon + β·KL

**Topologias configuráveis** (em YAML, não em código):

| Nome | hidden_dimensions |
|------|------------------|
| small | [256, 128] |
| medium | [512, 256] |
| deep | [512, 256, 128] |

Consulte [`docs/architecture.md`](docs/architecture.md) para descrição completa.

---

## Estrutura do Repositório

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── mypy.ini
├── .pre-commit-config.yaml
├── configs/
│   ├── base.yaml
│   ├── smoke_test.yaml
│   ├── mnist_unsupervised.yaml
│   ├── fashion_mnist_unsupervised.yaml
│   ├── topology_selection_mnist.yaml
│   ├── topology_selection_fashion_mnist.yaml
│   ├── latent_dimensions_mnist.yaml
│   ├── latent_dimensions_fashion_mnist.yaml
│   ├── mnist_supervised_optional.yaml
│   ├── fashion_mnist_supervised_optional.yaml
│   ├── factorial_protocol_mnist.yaml
│   └── factorial_protocol_fashion_mnist.yaml
├── docs/                         (documentação detalhada)
├── src/
│   └── vae_project/
│       ├── cli.py                (interface de linha de comando)
│       ├── config/               (carregamento e validação de YAML)
│       ├── data/                 (datasets, divisões, loaders)
│       ├── models/               (VAE, cabeça supervisionada, componentes)
│       ├── training/             (trainer, losses, early stopping, checkpointing)
│       ├── evaluation/           (extração latente, métricas)
│       ├── visualization/        (gráficos de treino, reconstrução, espaço latente)
│       ├── projections/          (PCA, UMAP, t-SNE, subsampling)
│       ├── experiments/          (serviço de execução individual, seleção de topologia,
│       │                          comparação de dims, protocolo fatorial)
│       ├── reporting/            (manifesto, sumário, relatório)
│       └── utils/                (device, paths, logging, timing, reproducibility)
├── tests/
│   ├── conftest.py
│   ├── unit/                     (testes unitários rápidos)
│   └── integration/              (testes de integração com dados sintéticos)
├── scripts/
│   ├── setup_windows.ps1
│   ├── run_smoke_test.ps1
│   ├── run_mnist_pipeline.ps1
│   ├── run_fashion_mnist_pipeline.ps1
│   ├── run_factorial_protocol_mnist.ps1
│   └── run_factorial_protocol_fashion_mnist.ps1
├── data/                         (.gitkeep — datasets baixados em runtime)
└── artifacts/                    (.gitkeep — artefatos gerados em runtime)
```

---

## Pré-requisitos

- Windows 10/11 (64-bit)
- Python 3.11 ou superior: https://python.org
- PowerShell 5.1 ou superior (incluído no Windows)
- Acesso à internet (primeiro download dos datasets)
- GPU NVIDIA com CUDA (opcional, CPU é obrigatório e suficiente)

---

## Instalação no Windows

### 1. Clone o repositório

```powershell
git clone https://github.com/joaopaulo164/vae-project.git
cd vae-project
```

### 2. Execute o script de setup

```powershell
.\scripts\setup_windows.ps1
```

### 3. Ative o ambiente virtual

```powershell
.venv\Scripts\Activate.ps1
```

> Se aparecer erro de política de execução:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Entenda as implicações antes de executar.

### 4. Atualize o pip e instale as dependências

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 5. Instale o PyTorch (CPU — padrão)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 6. Para PyTorch com CUDA (GPU NVIDIA)

Consulte o seletor oficial em https://pytorch.org para o comando específico
à sua versão de CUDA. Exemplo para CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 7. Verifique a instalação

```powershell
python -c "import vae_project; print(vae_project.__version__)"
python -m vae_project.cli smoke-test
```

Consulte [`docs/installation.md`](docs/installation.md) para instruções detalhadas.

---

## Execução Rápida (Smoke Test)

O smoke test usa **dados sintéticos** — não requer download nem GPU:

```powershell
# Via CLI
python -m vae_project.cli smoke-test

# Via script PowerShell
.\scripts\run_smoke_test.ps1

# Via pytest
python -m pytest tests/ -m "unit or integration" -v
```

> Os resultados do smoke test são identificados como `[QUICK MODE]` e
> **não devem ser reportados como resultados científicos**.

---

## Classificação dos Pipelines

| # | Etapa | Comando | Papel |
|---|-------|---------|-------|
| 1 | Execução individual | `run-pipeline` | Serviço técnico: uma topologia, uma dimensão, um modo; gera o conjunto completo de evidências |
| 2 | Seleção especializada de topologias | `select-topology` | Compara topologias para **uma dimensão fixa**; análise isolada |
| 3 | Comparação especializada de dimensões | `compare-latent-dims` | Compara dimensões para **uma topologia fixa**; análise isolada |
| 4 | **Protocolo fatorial principal** | `run-factorial-protocol` | **Protocolo científico principal**: 3 topologias × 4 dimensões, seleção, avaliação final e extensão supervisionada |
| 5 | Avaliação final | (fase do protocolo) | Avalia **somente a vencedora** no conjunto de teste, uma única vez |
| 6 | Extensão supervisionada | (fase do protocolo) | Treina um modelo novo com a mesma topologia e dimensão vencedoras |

Todos os comandos usam o mesmo serviço Python
(`execute_single_experiment`, em
[`experiments/single_experiment.py`](src/vae_project/experiments/single_experiment.py)).
O protocolo fatorial **não** invoca outros comandos de CLI por subprocesso.

---

## Execução Individual

`run-pipeline` executa **uma única configuração** (uma topologia, uma dimensão
latente, supervisionada ou não) e gera treinamento, reconstruções, amostras
geradas, espaço latente, tabelas, manifesto e resumo. Ele deixou de ser descrito
como o experimento científico principal: é o serviço técnico reutilizado pelos
demais fluxos.

```powershell
python -m vae_project.cli run-pipeline --config configs/mnist_unsupervised.yaml

python -m vae_project.cli run-pipeline --config configs/fashion_mnist_unsupervised.yaml
```

Também há os scripts `.\scripts\run_mnist_pipeline.ps1` e
`.\scripts\run_fashion_mnist_pipeline.ps1`. Para o protocolo principal, use
`.\scripts\run_factorial_protocol_mnist.ps1` e
`.\scripts\run_factorial_protocol_fashion_mnist.ps1`.

**Artefatos gerados:**
- `artifacts/mnist/unsupervised/latent_16/run_YYYYMMDD_HHMMSS_/`
  - `checkpoints/best_model.pt`
  - `figures/` (curvas de treino, reconstruções, espaço latente)
  - `tables/` (histórico CSV, métricas, distribuição de classes)
  - `latent_data/` (representações latentes CSV/NPZ)
  - `run_summary.md`
  - `config_resolved.yaml`
  - `environment.json`

Consulte [`docs/execution.md`](docs/execution.md) para mais exemplos.

---

## Seleção de Topologia

Treina três topologias (small, medium, deep) e seleciona a de menor
validation total loss. **O conjunto de teste não é utilizado.**

```powershell
python -m vae_project.cli select-topology --config configs/topology_selection_mnist.yaml

python -m vae_project.cli select-topology --config configs/topology_selection_fashion_mnist.yaml
```

**Resultado:** `topology_selection_results.csv` ordenado por `best_val_total_loss`.

Consulte [`docs/experimentation.md`](docs/experimentation.md) para o protocolo completo.

---

## Comparação de Dimensões Latentes

Mantém a topologia selecionada (`medium`) constante e varia a dimensão latente
(2, 8, 16, 32). Cada dimensão é um **treinamento independente**: um VAE novo,
com otimizador, scheduler e early stopping recriados e o seed reaplicado.
Todos os demais parâmetros (divisão treino/validação, seed, hiperparâmetros,
topologia, parâmetros das projeções e amostragem) são idênticos.

```powershell
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_mnist.yaml

python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_fashion_mnist.yaml
```

A lista de dimensões fica **somente** em `latent_dimension_comparison.dimensions`.
O campo `model.latent_dimension` continua sendo um único inteiro.

Para cada dimensão, o comando executa o fluxo completo do serviço comum:
treinamento, validação, seleção do melhor checkpoint (menor `val_total_loss`),
recarga desse checkpoint, curvas de treinamento, reconstruções, amostras geradas,
extração do `latent_mean`, figuras do espaço latente, métricas subsidiárias,
tabelas, manifesto e resumo da execução. Ao final, consolida todas as dimensões.
Este comando é uma análise especializada (topologia fixa); o protocolo científico
integral é o [`run-factorial-protocol`](#protocolo-fatorial-principal).

| Dimensão | Visualização gerada |
|----------|---------------------|
| 2 | Gráfico direto `latent_mean_0` × `latent_mean_1` (sem PCA/UMAP/t-SNE) |
| 8, 16, 32 | PCA, UMAP, t-SNE e variância explicada do PCA (PNG + SVG) |

Pontos importantes:

- O experimento é **não supervisionado** (`supervised_component.enabled: false`).
  Os rótulos são usados apenas depois do treino, para colorir as figuras e
  calcular métricas subsidiárias. O experimento supervisionado é executado
  manualmente, com uma configuração independente, depois de escolher a dimensão.
- Há um único seed (`global_seed: 42`); não há lista de seeds.
- Somente o PCA possui variância explicada. UMAP e t-SNE não a possuem.
- A análise visual é a evidência principal; as métricas são subsidiárias.
- Em CPU, o tempo de execução pode ser longo (quatro treinamentos completos mais
  UMAP e t-SNE para cada dimensão maior que 2). A primeira execução do UMAP
  também compila código com numba, o que adiciona cerca de 30 segundos.
- Uma falha em uma dimensão não interrompe as demais. O status de cada dimensão
  (`completed`, `partial` ou `failed`) aparece na tabela consolidada.

**Resultado:** `latent_dimension_comparison.csv` e
`latent_dimension_comparison_summary.md`, mais um diretório `latent_dim_N/` por
dimensão. Veja [Localização dos Artefatos](#localização-dos-artefatos) e
[`docs/experimentation.md`](docs/experimentation.md).

---

## Protocolo Fatorial Principal

`run-factorial-protocol` é o **protocolo científico principal**. Ele automatiza
todo o fluxo:

```
small, medium, deep  ×  2D, 8D, 16D, 32D
        ↓
12 configurações não supervisionadas por dataset
        ↓
seleção da menor validation total loss
        ↓
avaliação no teste somente da combinação vencedora
        ↓
extensão supervisionada (mesma topologia e dimensão, modelo novo)
        ↓
avaliação da extensão no teste e comparação entre os modos
        ↓
relatório, índice de evidências e manifesto global
```

Comandos (PowerShell, uma linha):

```powershell
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml

python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_fashion_mnist.yaml
```

Retomada de uma rodada interrompida (use o diretório `run_...` da rodada):

```powershell
python -m vae_project.cli run-factorial-protocol --config configs/factorial_protocol_mnist.yaml --resume artifacts/mnist/unsupervised/factorial_protocol/run_YYYYMMDD_HHMMSS_identificador
```

Pontos principais:

- **3 topologias × 4 dimensões = 12 configurações por dataset**, ou seja,
  **24 treinamentos de grade** nos dois datasets (MNIST e Fashion-MNIST), mais um
  treinamento supervisionado por dataset.
- Cada combinação cria um VAE novo, com otimizador, scheduler e early stopping
  novos e o seed reaplicado; nenhum peso é reaproveitado.
- Somente topologia, `hidden_dimensions` e dimensão latente variam. Há **um único
  seed** (`global_seed: 42`; divisões e `random_state` das projeções também 42).
- A grade é **não supervisionada**: os rótulos só colorem figuras e alimentam
  métricas subsidiárias depois do treino.
- A seleção usa **somente a validação**: menor `validation_total_loss`, com
  desempates determinísticos (reconstruction loss, número de parâmetros, dimensão
  latente, nome da topologia). Execuções com falha, incompletas, sem checkpoint
  ou com valores não finitos são excluídas.
- **Somente a vencedora** é avaliada no teste, **uma vez**, com o melhor
  checkpoint (sem retreinar). O teste não altera topologia, dimensão,
  hiperparâmetros, época ou decisão de seleção.
- A **extensão supervisionada** treina um modelo novo com a mesma configuração
  estrutural e de treino; só mudam a cabeça classificadora, a classification loss
  e o tipo do experimento. Com `execution_mode: manual` ela não treina: apenas
  gera `generated_configs/supervised_selected_configuration.yaml` e informa o
  comando de execução manual.
- A `total_loss` supervisionada **não** é comparável à não supervisionada (a
  composição difere); a comparação é feita por reconstrução, KL, figuras,
  métricas subsidiárias e desempenho classificatório.
- **Retomada** (`--resume`): valida hash da configuração, dataset e artefatos
  reais (status, checkpoint, métricas, resumo, manifesto, evidências mínimas). Um
  diretório existente não é suficiente para considerar a combinação concluída.
- **Prevenção de sobrescrita:** cada execução cria um `run_YYYYMMDD_HHMMSS_<id>/`
  novo e exclusivo; `--output-dir` é o diretório *base* e não habilita
  sobrescrita (o protocolo não possui `--overwrite`).
- **Duração:** em CPU a rodada é muito longa (12 treinamentos completos, mais
  UMAP e t-SNE para as dimensões 8, 16 e 32, mais o treinamento supervisionado).
  CUDA é opcional (`runtime.device: auto`).
- Somente o PCA possui variância explicada. A análise visual é a evidência
  principal; as métricas são subsidiárias.
- Resultados de testes e de smoke tests usam dados **sintéticos**, ficam em
  diretórios `smoke_test`/`synthetic` e **não** são resultados científicos.

Detalhes: [`docs/experimentation.md`](docs/experimentation.md),
[`docs/configuration.md`](docs/configuration.md) e
[`docs/execution.md`](docs/execution.md).

---

## Componente Supervisionado Opcional

O componente supervisionado é **desabilitado por padrão** e deve ser
habilitado explicitamente. Resultados supervisionados e não supervisionados
são armazenados em diretórios separados.

Para habilitar, use o config com `supervised_component.enabled: true`:

```powershell
python -m vae_project.cli run-pipeline --config configs/mnist_supervised_optional.yaml
```

> **Importante:** Os rótulos influenciam o treinamento apenas quando
> `supervised_component.enabled: true`. No modo não supervisionado (padrão),
> os rótulos são utilizados **somente** para colorir visualizações.

Consulte [`docs/experimentation.md`](docs/experimentation.md) para detalhes.

---

## Testes Automatizados

```powershell
# Todos os testes (unit + integration, sem GPU, sem downloads)
python -m pytest

# Apenas testes unitários
python -m pytest -m unit

# Apenas integração
python -m pytest -m integration

# Com cobertura
python -m pytest --cov=vae_project --cov-report=term-missing

# Relatório HTML de cobertura
python -m pytest --cov=vae_project --cov-report=html
```

> Os testes usam **apenas dados sintéticos** e não requerem internet.

Consulte [`docs/testing.md`](docs/testing.md) para descrição de cada teste.

---

## Qualidade de Código

```powershell
# Lint (Ruff)
python -m ruff check .

# Formatação (Black — verificação apenas)
python -m black --check .

# Aplicar formatação
python -m black .

# Verificação de tipos (mypy)
python -m mypy src

# Segurança (Bandit)
python -m bandit -r src

# Todos os hooks pre-commit
python -m pre_commit run --all-files
```

---

## Localização dos Artefatos

Cada execução cria um diretório único com timestamp:

```
artifacts/
├── mnist/
│   ├── unsupervised/
│   │   └── latent_16/
│   │       └── run_20240101_120000/
│   │           ├── checkpoints/best_model.pt
│   │           ├── figures/
│   │           ├── tables/
│   │           ├── latent_data/
│   │           ├── logs/experiment.log
│   │           ├── run_summary.md
│   │           ├── config_resolved.yaml
│   │           └── environment.json
│   └── supervised/
└── fashion_mnist/
    ├── unsupervised/
    └── supervised/
```

Execuções anteriores **nunca são sobrescritas** (timestamps únicos).

A comparação de dimensões latentes usa um diretório próprio por dataset e tipo
de experimento. **Cada execução cria um diretório exclusivo**
`run_YYYYMMDD_HHMMSS_<id>/`, com um subdiretório por dimensão:

```
artifacts/mnist/unsupervised/latent_dimension_comparison/
└── run_20260101_120000_a1b2c3/
    ├── latent_dimension_comparison.csv
    ├── latent_dimension_comparison_summary.md
    ├── latent_dim_2/
    ├── latent_dim_8/
    ├── latent_dim_16/
    └── latent_dim_32/
        ├── checkpoints/best_model.pt
        ├── figures/{training,reconstructions,generated_samples,latent_space}/
        ├── latent_data/
        ├── tables/
        ├── logs/experiment.log
        ├── config_resolved.yaml
        ├── metrics.json
        ├── run_summary.md
        └── artifact_manifest.json
```

O protocolo fatorial grava em
`artifacts/<dataset>/unsupervised/factorial_protocol/run_YYYYMMDD_HHMMSS_<id>/`,
com `shared/`, `grid/<topologia>/latent_dim_<N>/`, `selection/`,
`final_evaluation/`, `supervised_extension/`, `generated_configs/`,
`protocol_state.json`, `protocol_manifest.json`, `protocol_summary.md` e
`scientific_evidence_index.md`. A estrutura completa está em
[`docs/evidence_catalog.md`](docs/evidence_catalog.md).

Reexecutar o comando **não** sobrescreve execuções anteriores. `--output-dir` é
o diretório *base* (a execução cria o `run_...` dentro dele); apontá-lo para um
`run_...` existente é rejeitado com uma mensagem de conflito, a menos que se
passe `--overwrite` explicitamente. Veja [`docs/execution.md`](docs/execution.md).

O `best_model.pt` contém pesos, estado do otimizador, do scheduler, do early
stopping e a configuração resolvida; permite retomar o treinamento, e
checkpoints antigos (só pesos) continuam válidos para gerar evidências. Veja
[`docs/reproducibility.md`](docs/reproducibility.md).

### Dados sintéticos e idioma

Testes e smoke test usam dados **sintéticos**; seus diretórios contêm
`smoke_test` ou `synthetic` e seus relatórios declaram que não são resultados
científicos. A comparação científica usa somente MNIST e Fashion-MNIST.

Convenção de idioma: a documentação (README e `docs/`) é em **português**;
código, identificadores, docstrings, logs e mensagens técnicas são em
**inglês**; os relatórios gerados pelo código (`run_summary.md`,
`latent_dimension_comparison_summary.md`, `selection_summary.md`,
`protocol_summary.md`, `comparison_summary.md` e `scientific_evidence_index.md`)
são em **inglês**.

---

## Análise Visual e Métricas Subsidiárias

### Evidência principal: análise visual

A interpretação do espaço latente deve ser baseada primariamente na
análise visual dos gráficos de projeção. As visualizações permitem
observar diretamente:

- Formação aparente de clusters
- Separação entre rótulos
- Sobreposição entre classes
- Compacidade e dispersão
- Outliers e classes frequentemente misturadas

### Métricas subsidiárias

As métricas quantitativas (Silhouette Score, Calinski-Harabasz, Davies-Bouldin,
trustworthiness, classificador linear) são **evidências de suporte**, não
conclusões primárias.

**Regras obrigatórias:**
- Nenhuma métrica isolada é prova definitiva de clusters
- ARI e NMI devem ser tratados como comparações exploratórias
- O Silhouette Score é afetado por subamostragem — verificar `was_subsampled`
- O classificador linear mede separabilidade linear, não presença de clusters

Consulte [`docs/results_interpretation.md`](docs/results_interpretation.md)
para o roteiro completo de análise.

---

## Limitações de PCA, UMAP e t-SNE

| Método | Fornece variância explicada? | Preserva estrutura global? | Nota |
|--------|------------------------------|---------------------------|------|
| PCA | ✓ Sim (linear) | Parcialmente | Variância linear; não mede separação de classes |
| UMAP | ✗ Não | Parcialmente (local) | Parâmetros afetam fortemente o resultado |
| t-SNE | ✗ Não | Apenas local | Tamanhos e distâncias entre clusters não são interpretáveis |

**Nunca reportar variância explicada para UMAP ou t-SNE.**

Consulte [`docs/scientific_methodology.md`](docs/scientific_methodology.md)
para a discussão metodológica completa.

---

## Reprodutibilidade

O projeto implementa reprodutibilidade por:

- Seed global configurável (`reproducibility.global_seed`)
- Seeds para Python, NumPy, PyTorch CPU e CUDA
- Modo determinístico opcional (`reproducibility.deterministic_mode`)
- Registro do ambiente completo (`environment.json`)
- Configuração resolvida salva por execução (`config_resolved.yaml`)
- Hash do commit Git (quando disponível)

> **Limitação:** Reprodutibilidade absoluta entre diferentes hardware,
> sistemas operacionais, versões de CUDA ou bibliotecas **não é garantida**.

Consulte [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Mapa da Documentação

| Documento | Conteúdo |
|-----------|---------|
| [`docs/architecture.md`](docs/architecture.md) | Arquitetura do VAE, componentes, topologias |
| [`docs/configuration.md`](docs/configuration.md) | Todas as opções de configuração YAML |
| [`docs/datasets.md`](docs/datasets.md) | Datasets, divisões, anti-leakage |
| [`docs/installation.md`](docs/installation.md) | Instalação detalhada Windows |
| [`docs/execution.md`](docs/execution.md) | Comandos de execução e exemplos |
| [`docs/experimentation.md`](docs/experimentation.md) | Protocolo experimental completo |
| [`docs/evaluation.md`](docs/evaluation.md) | Avaliação, métricas, visualizações |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seeds, determinismo, limitações |
| [`docs/scientific_methodology.md`](docs/scientific_methodology.md) | Metodologia científica |
| [`docs/results_interpretation.md`](docs/results_interpretation.md) | Roteiro de análise |
| [`docs/article_evidence_guide.md`](docs/article_evidence_guide.md) | Guia para artigo científico |
| [`docs/testing.md`](docs/testing.md) | Testes automatizados |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Solução de problemas |
| [`docs/project_structure.md`](docs/project_structure.md) | Estrutura detalhada do projeto |

---

## Solução de Problemas

### ModuleNotFoundError: No module named 'vae_project'

```powershell
pip install -e ".[dev]"
```

### CUDA requested but not available

Edite o arquivo de configuração e defina `runtime.device: cpu` ou `auto`.

### Execution Policy error (PowerShell)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### num_workers > 0 causa erros no Windows

O valor padrão `num_workers: 0` é seguro para Windows. Não altere sem necessidade.

### Download falhou / sem conexão

Na primeira execução é necessária conexão. Após o download, os dados ficam
em `data/` e podem ser reutilizados offline.

Consulte [`docs/troubleshooting.md`](docs/troubleshooting.md) para mais casos.

---

## Licença e Citação

Este projeto é distribuído sob a licença [MIT](LICENSE).

Para citar este software em publicações científicas, preencha os campos
`[REQUIRED]` no arquivo [`CITATION.cff`](CITATION.cff) com seus dados
reais (nome, instituição, ORCID) antes de publicar.

```bibtex
@software{vae_project,
  title = {VAE Project: Variational Autoencoder Study for MNIST and Fashion-MNIST},
  author = {Mendes, João Paulo dos Santos Mendes},
  year = {2026},
  url = {https://github.com/joaopaulo164/vae-project},
  version = {1.0.0}
}
```
