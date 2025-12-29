# Ablation report: KoVAE gamma=0 vs gamma=0.1

## Resumen rápido
- Figura comparativa: reports/figures/ablation_long_mae_by_horizon.png

## MAE por horizonte
|Horizon|MAE gamma=0.0|MAE gamma=0.1|
|---:|---:|---:|
|1|0.003009867700500607|1.7150429621988916|

## Observaciones y recomendaciones
- Los valores de pérdida durante el entrenamiento difieren en escala; usar MAE/RMSE en unidades reales para decisiones.
- Enmascarar celdas de varianza casi nula para interpretar R².
- Revisar mapas de error y las 20 peores celdas (ya generadas en `data/processed/kovae_worst_cells_examples.csv` si existe).
- Evaluar desempeño en eventos extremos (top 5% precipitación).
