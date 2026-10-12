# Projeto VAE — Variational Autoencoders para MNIST e Fashion-MNIST

Um projeto científico completo, reprodutível e executável para estudo de
**Variational Autoencoders (VAE)** com os datasets MNIST e Fashion-MNIST.

> **Leia antes de executar:** Os resultados produzidos pelo projeto devem ser
> interpretados visualmente antes de recorrer a métricas quantitativas.
> Consulte a seção [Análise Visual e Métricas Subsidiárias](#análise-visual-e-métricas-subsidiárias).

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
9. [Pipelines de Experimento](#pipelines-de-experimento)
10. [Seleção de Topologia](#seleção-de-topologia)
11. [Comparação de Dimensões Latentes](#comparação-de-dimensões-latentes)
12. [Componente Supervisionado Opcional](#componente-supervisionado-opcional)
13. [Testes Automatizados](#testes-automatizados)
14. [Qualidade de Código](#qualidade-de-código)
15. [Localização dos Artefatos](#localização-dos-artefatos)
16. [Análise Visual e Métricas Subsidiárias](#análise-visual-e-métricas-subsidiárias)
17. [Limitações de PCA, UMAP e t-SNE](#limitações-de-pca-umap-e-t-sne)
18. [Reprodutibilidade](#reprodutibilidade)
19. [Mapa da Documentação](#mapa-da-documentação)
20. [Solução de Problemas](#solução-de-problemas)
21. [Licença e Citação](#licença-e-citação)

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
│   └── fashion_mnist_supervised_optional.yaml
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
│       ├── experiments/          (runner, seleção de topologia, comparação de dims)
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
│   └── run_fashion_mnist_pipeline.ps1
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

## Pipelines de Experimento

### MNIST (Experimento Principal — não supervisionado)

```powershell
# Pipeline completo
python -m vae_project.cli run-pipeline --config configs/mnist_unsupervised.yaml

# Via script
.\scripts\run_mnist_pipeline.ps1
```

### Fashion-MNIST

```powershell
python -m vae_project.cli run-pipeline --config configs/fashion_mnist_unsupervised.yaml

# Via script
.\scripts\run_fashion_mnist_pipeline.ps1
```

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

Mantém a topologia selecionada constante e varia a dimensão latente
(2, 8, 16, 32). Todos os demais parâmetros são mantidos idênticos.

```powershell
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_mnist.yaml

python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_fashion_mnist.yaml
```

**Resultado:** `latent_dimension_comparison.csv` com métricas por dimensão.

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
