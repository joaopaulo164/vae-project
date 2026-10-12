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
Resultados marcados com [QUICK MODE] não são científicos.
