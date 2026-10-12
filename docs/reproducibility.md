# Avaliação

## Extração de Representações Latentes

Módulo: \evaluation/latent_extraction.py\
Extrai latent_mean (determinístico), latent_log_variance e latent_sample.
latent_mean é preferido para projeções determinísticas.

## Métricas de Reconstrução

- MSE médio por pixel
- BCE médio por pixel
Módulo: \evaluation/reconstruction_metrics.py\

## Métricas de Clustering (subsidiárias)

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
Módulo: \evaluation/clustering_metrics.py\

## Classificador Linear (subsidiário)

Regressão logística sobre latent_mean.
Treino: split de treino. Seleção: validação. Teste: apenas na fase final.
Módulo: \evaluation/classification_metrics.py\
"@
Set-Content -Path "docs/evaluation.md" -Value # Protocolo Experimental

## Fase A: Seleção de Topologia

Treina small, medium e deep com mesma latent_dim, seed, lr, etc.
Seleciona topologia com menor val_total_loss.
Comando: \select-topology\

## Fase B: Comparação de Dimensões Latentes

Mantém topologia selecionada. Varia apenas latent_dimension: [2, 8, 16, 32].
Parâmetros constantes: dataset, seed, lr, batch_size, max_epochs, loss, splits.
Comando: \compare-latent-dims\

## Fase C: Avaliação Final

Carrega melhor checkpoint. Avalia UMA VEZ no conjunto de teste.
Resultados claramente marcados como métricas de teste.
Módulo: \experiments/final_evaluation.py\

## Experimento Supervisionado (opcional)

Habilitado por supervised_component.enabled: true.
Resultados armazenados em diretório separado (supervised/).
Nunca misturar com resultados do experimento não supervisionado.

# reproducibility.md  
# Protocolo Experimental

## Fase A: Seleção de Topologia

Treina small, medium e deep com mesma latent_dim, seed, lr, etc.
Seleciona topologia com menor val_total_loss.
Comando: \select-topology\

## Fase B: Comparação de Dimensões Latentes

Mantém topologia selecionada. Varia apenas latent_dimension: [2, 8, 16, 32].
Parâmetros constantes: dataset, seed, lr, batch_size, max_epochs, loss, splits.
Comando: \compare-latent-dims\

## Fase C: Avaliação Final

Carrega melhor checkpoint. Avalia UMA VEZ no conjunto de teste.
Resultados claramente marcados como métricas de teste.
Módulo: \experiments/final_evaluation.py\

## Experimento Supervisionado (opcional)

Habilitado por supervised_component.enabled: true.
Resultados armazenados em diretório separado (supervised/).
Nunca misturar com resultados do experimento não supervisionado. = @"
# Reprodutibilidade

## Seeds Configurados

- Python random: random.seed(seed)
- NumPy: np.random.seed(seed)
- PyTorch CPU: torch.manual_seed(seed)
- PyTorch CUDA: torch.cuda.manual_seed_all(seed)

Módulo: \utils/reproducibility.py\

## Modo Determinístico

Habilitado por: reproducibility.deterministic_mode: true
Ativa torch.use_deterministic_algorithms(True).
Pode causar erros se não houver kernel determinístico disponível.

## Metadados Registrados por Execução

- environment.json: versões de todas as bibliotecas, OS, hardware, git hash
- config_resolved.yaml: configuração completa resolvida
- run_summary.md: resumo da execução

## Limitações

Reprodutibilidade bit-a-bit entre diferentes:
- Hardware (CPU vs GPU)
- Versões de CUDA
- Versões de bibliotecas
NÃO é garantida, mesmo com seeds idênticos.
