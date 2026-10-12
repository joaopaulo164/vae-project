# Instalação no Windows

## Requisitos

- Windows 10/11 (64-bit)
- Python 3.11 ou superior
- PowerShell 5.1 ou superior

## Passos

### 1. Criar ambiente virtual

\\\powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
\\\

### 2. Instalar dependências

\\\powershell
python -m pip install --upgrade pip
pip install -e '.[dev]'
\\\

### 3. Instalar PyTorch (CPU)

\\\powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
\\\

### 4. Instalar PyTorch com CUDA

Consulte https://pytorch.org para o comando correto para sua versão CUDA.

### 5. Verificar instalação

\\\powershell
python -c 'import vae_project; print(vae_project.__version__)'
python -m vae_project.cli smoke-test
\\\
"@
Set-Content -Path "docs/installation.md" -Value # Datasets

## MNIST

- 10 classes: dígitos 0–9
- Treino oficial: 60.000 amostras
- Teste oficial: 10.000 amostras
- Imagens: 28x28 pixels, escala de cinza
- Carregado via: torchvision.datasets.MNIST

## Fashion-MNIST

- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Treino oficial: 60.000 amostras
- Teste oficial: 10.000 amostras
- Imagens: 28x28 pixels, escala de cinza
- Carregado via: torchvision.datasets.FashionMNIST

## Normalização

Imagens normalizadas para [0, 1] por ToTensor() (divisão por 255).
Não é aplicada normalização z-score para manter compatibilidade com
a saída Sigmoid do decoder e a reconstruction loss BCE.

## Divisão Treino/Validação/Teste

- Conjunto de treino oficial ? divisão estratificada 90%/10%
- Seed configurável em splits.seed (padrão: 42)
- Validação: seleção de modelo e early stopping
- Teste: avaliação final única

## Prevenção de Data Leakage

O conjunto de teste NUNCA é usado para:
- Selecionar topologia
- Comparar dimensões latentes
- Ajustar hiperparâmetros
- Early stopping ou seleção de checkpoint

## Quick Mode

quick_mode: true usa subconjunto reduzido para smoke tests.
Resultados marcados com [QUICK MODE] não são científicos.

# execution.md
# Datasets

## MNIST

- 10 classes: dígitos 0–9
- Treino oficial: 60.000 amostras
- Teste oficial: 10.000 amostras
- Imagens: 28x28 pixels, escala de cinza
- Carregado via: torchvision.datasets.MNIST

## Fashion-MNIST

- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Treino oficial: 60.000 amostras
- Teste oficial: 10.000 amostras
- Imagens: 28x28 pixels, escala de cinza
- Carregado via: torchvision.datasets.FashionMNIST

## Normalização

Imagens normalizadas para [0, 1] por ToTensor() (divisão por 255).
Não é aplicada normalização z-score para manter compatibilidade com
a saída Sigmoid do decoder e a reconstruction loss BCE.

## Divisão Treino/Validação/Teste

- Conjunto de treino oficial ? divisão estratificada 90%/10%
- Seed configurável em splits.seed (padrão: 42)
- Validação: seleção de modelo e early stopping
- Teste: avaliação final única

## Prevenção de Data Leakage

O conjunto de teste NUNCA é usado para:
- Selecionar topologia
- Comparar dimensões latentes
- Ajustar hiperparâmetros
- Early stopping ou seleção de checkpoint

## Quick Mode

quick_mode: true usa subconjunto reduzido para smoke tests.
Resultados marcados com [QUICK MODE] não são científicos. = @"
# Execução

## Comandos CLI

\\\powershell
# Validar configuração
python -m vae_project.cli validate-config --config configs/mnist_unsupervised.yaml

# Treinar (pipeline completo)
python -m vae_project.cli run-pipeline --config configs/mnist_unsupervised.yaml

# Selecionar topologia
python -m vae_project.cli select-topology --config configs/topology_selection_mnist.yaml

# Comparar dimensões latentes
python -m vae_project.cli compare-latent-dims --config configs/latent_dimensions_mnist.yaml

# Smoke test (sem downloads)
python -m vae_project.cli smoke-test
\\\

## Artefatos Gerados

Cada execução cria um diretório único em:
artifacts/{dataset}/{unsupervised|supervised}/latent_{dim}/run_{timestamp}/
