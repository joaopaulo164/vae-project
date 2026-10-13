# Interpretação dos Resultados

Este documento fornece um roteiro para análise dos resultados **após a execução
real dos experimentos**. Nenhuma conclusão deve ser preenchida antes de analisar
as figuras e tabelas geradas.

## Princípio Fundamental

> A análise visual é a evidência primária. As métricas quantitativas são
> subsidiárias e jamais devem ser a única base de uma conclusão.

---

## 1. Análise do Espaço Latente — Perguntas-Guia

### Sobre formação de clusters

- Há agrupamentos visualmente identificáveis nas projeções?
- Os agrupamentos aparecem de forma consistente em PCA, UMAP e t-SNE?
- Ou apenas em um método? (Pode ser artefato do método de projeção.)
- Os clusters aparecem também no gráfico direto 2D (quando latent_dim=2)?

### Sobre separação entre classes

- Quais classes apresentam separação visual clara?
- Quais classes se sobrepõem visualmente?
- Quais pares de classes aparecem frequentemente misturados?
- A separação se mantém entre PCA, UMAP e t-SNE?

### Sobre compacidade e dispersão

- Quais classes formam grupos compactos?
- Quais classes apresentam alta dispersão (espalhadas pelo espaço)?
- Há outliers visíveis (pontos isolados longe do grupo)?

### Sobre diferenças entre métodos de projeção

- Quais clusters visíveis em t-SNE desaparecem no PCA?
- A separação aparente no UMAP é confirmada pelo PCA?
- As perguntas acima ajudam a identificar artefatos de projeção.

### Sobre variância explicada do PCA

- Qual percentual da variância linear é retido pelos 2 primeiros componentes?
- Um percentual baixo indica que PCA pode não capturar toda a estrutura relevante.
- Um percentual alto indica boa representação linear, mas não implica separação de classes.

---

## 2. Comparação MNIST vs Fashion-MNIST

Após rodar ambos os pipelines, compare:

- As classes do MNIST apresentam maior separação visual do que Fashion-MNIST?
- Quais classes do Fashion-MNIST mais se sobrepõem? (Ex.: Shirt vs T-shirt/top)
- A variância explicada pelo PCA difere entre os datasets?
- A trustworthiness das projeções difere entre os datasets?

---

## 3. Impacto da Dimensão Latente

Fontes: `latent_dimension_comparison.csv`, `latent_dimension_comparison_summary.md` e
as figuras de cada `latent_dim_N/figures/latent_space/`.

- Como o loss de reconstrução varia com a dimensão latente?
- Como o KL divergence varia?
- Dimensões maiores produzem separação visual melhor?
- A dimensão 2 (visualizável diretamente) preserva estrutura similar às demais?
- As projeções PCA, UMAP e t-SNE das dimensões 8, 16 e 32 concordam entre si?

Cuidados específicos:

- A dimensão 2 é desenhada diretamente (`latent_mean_0` × `latent_mean_1`). As
  demais são **projetadas** para 2D; formas e distâncias não são diretamente
  comparáveis com o gráfico 2D direto.
- A `validation_total_loss` do resumo consolidado indica apenas qual dimensão teve a
  menor loss de validação. Ela **não** indica qual dimensão forma melhores
  agrupamentos; isso exige a análise visual.
- Somente o PCA possui variância explicada. Não reporte esse valor para UMAP ou t-SNE.
- Projeções de dimensões maiores usam uma amostra estratificada (os títulos indicam
  "sample of N"). Os índices estão em `projection_sample_indices.csv`.
- Dimensões com status `partial` ou `failed` têm evidências incompletas; consulte
  `error_summary` e o `run_summary.md` da dimensão antes de comparar.
- O experimento é não supervisionado e usa um único seed (42). Diferenças pequenas
  entre dimensões podem ser variação de treinamento e não devem ser tratadas como
  conclusivas sem repetição.

---

## 4. Componente Supervisionado (quando habilitado)

- A separação visual melhorou em relação ao VAE não supervisionado?
- O loss de reconstrução foi afetado negativamente?
- A melhora na separação visual é proporcional ao custo em reconstrução?

---

## 5. Métricas Subsidiárias — Interpretação

| Métrica | Interpretação correta | Armadilha a evitar |
|---------|----------------------|-------------------|
| Silhouette Score | Compacidade e separação das classes no espaço latente | Não prova existência de clusters naturais |
| Calinski-Harabasz | Razão variância entre/intra classes | Favorece classes bem esféricas |
| Davies-Bouldin | Similaridade média entre classes | Menor é melhor, mas não mede clusters naturais |
| Trustworthiness | Preservação de vizinhança local na projeção | Apenas para projeções, não para o espaço original |
| PCA Explained Variance | Variância linear retida | Variância ≠ separação de classes |
| Linear Classifier Accuracy | Separabilidade linear no espaço latente | Alta acurácia ≠ clusters naturais compactos |

---

## 6. Distinções Importantes para Escrita Científica

| Tipo | Exemplo correto |
|------|----------------|
| Observação visual | "Na projeção t-SNE, pontos da classe X formam um grupo visualmente separado de Y." |
| Métrica quantitativa | "O Silhouette Score de 0.XX (n=XXXX) indica moderada compacidade das classes no espaço latente." |
| Interpretação | "A separação visual entre X e Y sugere que o VAE capturou características discriminativas." |
| Limitação | "A separação aparente no t-SNE não foi confirmada no PCA, podendo ser artefato do método." |
| Hipótese | "Investigações futuras deveriam verificar se a sobreposição entre Shirt e T-shirt/top persiste em dimensões superiores." |

---

## 7. O que NÃO fazer

- Não descrever resultados antes de executar os experimentos
- Não usar apenas uma métrica para concluir que há clusters
- Não reportar variância explicada para UMAP ou t-SNE
- Não comparar losses de configurações com reconstruction_loss_type diferente
- Não usar resultados de quick_mode como resultados científicos finais
- Não selecionar a configuração "melhor" após ver os resultados de teste
