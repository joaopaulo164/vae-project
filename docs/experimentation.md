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
Nunca misturar com resultados do experimento não supervisionado.
