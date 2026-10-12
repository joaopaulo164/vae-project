# Metodologia Científica

Este documento descreve o protocolo experimental do projeto VAE.

## Pergunta Experimental

O projeto investiga: *Em que medida um VAE não supervisionado organiza o espaço
latente de forma que reflita as categorias semânticas dos datasets MNIST e
Fashion-MNIST, e como essa organização varia com a dimensionalidade latente?*

## Protocolo de Dados

### Divisão treino/validação/teste

- O conjunto oficial de **treinamento** é dividido estratificadamente em:
  - **Treino** (90% por padrão): usado para atualizar os parâmetros do modelo.
  - **Validação** (10% por padrão): usado para seleção do melhor checkpoint e
    comparação de topologias/dimensões. **Nunca usado para atualizar parâmetros.**
- O conjunto oficial de **teste** é reservado para a avaliação final única.
  - **Nunca usado para selecionar modelo, topologia, dimensão ou hiperparâmetros.**

### Prevenção de data leakage

As fases experimentais garantem separação estrita:

| Fase | Conjunto usado | Decisão |
|------|---------------|---------|
| A — Seleção de topologia | Treino + Validação | Menor val_total_loss |
| B — Comparação de dimensões | Treino + Validação | Comparação por validação |
| C — Avaliação final | Teste (único) | Relatório final |

## VAE Loss — ELBO

O VAE maximiza o ELBO (Evidence Lower BOund):

```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
```

Minimizar o negativo do ELBO é equivalente a minimizar:

```
total_vae_loss = reconstruction_loss + β × kl_divergence_loss
```

### Reconstruction Loss

Implementada como BCE (padrão) para imagens em [0, 1]:

```
L_recon = -Σ [x·log(x̂) + (1-x)·log(1-x̂)]  (somado por pixel, média por batch)
```

**Justificativa:** BCE trata os pixels como probabilidades Bernoulli, compatível
com a saída Sigmoid do decoder. MSE está disponível como alternativa.

### KL Divergence

Forma fechada para q(z|x) = N(µ, diag(σ²)) e p(z) = N(0, I):

```
KL = -0.5 × Σ_j (1 + log σ²_j - µ²_j - σ²_j)
```

O parâmetro `beta` (padrão 1.0) pondera o KL. Qualquer valor ≠ 1.0 é
registrado como desvio do VAE padrão.

## Métodos de Projeção

### PCA

- Projeção linear que maximiza variância explicada
- Fornece `explained_variance_ratio` — proporção da variância total retida
- **Limitação:** Variância explicada linear ≠ separação entre classes

### UMAP

- Projeção não-linear baseada em manifold
- Preserva estrutura local de vizinhança
- **Não fornece variância explicada**
- Parâmetros afetam fortemente o resultado visual

### t-SNE

- Projeção não-linear estocástica
- Preserva apenas estrutura local
- **Não fornece variância explicada**
- Distâncias globais e tamanhos de clusters não são diretamente interpretáveis

## Análise Visual como Evidência Principal

A análise do espaço latente deve ser primariamente visual. As figuras geradas
permitem observar:

- Formação aparente de clusters
- Separação entre rótulos
- Sobreposição entre classes

As métricas quantitativas são **subsidiárias** — apoiam mas não substituem
a interpretação visual.

## Limitações Metodológicas

1. Reprodutibilidade entre hardware/OS distintos não é garantida
2. t-SNE não preserva distâncias globais
3. Métricas de clustering usam rótulos como pseudo-clusters — não como clusters descobertos
4. O classificador linear mede separabilidade linear, não estrutura de cluster
5. Quick mode usa dados insuficientes para conclusões científicas
