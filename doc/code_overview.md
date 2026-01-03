# Code Overview

Este documento resume la arquitectura y propósito de los principales componentes del proyecto de Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile.

## Flujo General

1. (Opcional) Descarga datos ERA5 2020: `src/utils/download_era5_small.py`.
2. Merge y preprocesamiento a formato combinado: `src/utils/merge_era5_small.py` genera NetCDF y CSV diario.
3. Fallback (cuando no hay ERA5): `src/utils/fallback_build_from_npy.py` crea un NetCDF sintético y CSV a partir de `precipitation_data.npy`.
4. Exploración y EDA: `notebooks/01A_Eda_spatiotemporal.ipynb`.
5. Modelado AE-DMD (Autoencoder + Dynamic Mode Decomposition): `notebooks/02_DL_DMD_Forecast.ipynb` y script reproducible `src/train_ae_dmd_v2.py` con MLflow.

## Componentes Clave

- `models/ae_dmd.py`: Implementa clase compuesta AE_DMD que entrena un autoencoder y aplica DMD sobre representaciones latentes para pronóstico.
- `models/ae_keras.py`: Arquitecturas de autoencoder basadas en Keras (densa o con LSTM si se activa la bandera `use_lstm`).
- `models/kovae.py`: Variante experimental VAE (KOpula/VAE) para explorar generación y regularización.
- `utils/mlflow_utils.py`: Configura tracking de MLflow (URI local y experimento).
- `utils/metrics.py`: Métricas de evaluación (MAE, RMSE, NSE, R2, etc.).
- `utils/download_era5_small.py`: Descarga mensual de ERA5 (total_precipitation) en bounding box de Chile.
- `utils/merge_era5_small.py`: Combina y resamplea los NetCDF mensuales a diario; genera dataset completo.
- `utils/fallback_build_from_npy.py`: Construye dataset procesado desde matriz pura .npy (sin metadatos espaciales reales) para desbloquear EDA.

## Datos

- `data/raw/precipitation_data.npy`: Matriz base (T x 30) usada como fallback; columnas S1..S30 = series sin metadatos.
- `data/raw/precipitation_test.csv`: Representación CSV de la misma matriz para compatibilidad rápida.
- `data/processed/era5_precipitation_chile_full.nc`: Dataset NetCDF combinado (real o sintético) con variables espaciales.
- `data/processed/era5_precipitation_chile_daily.csv`: CSV diario (lat, lon, date, total_precipitation).

## Concepto AE-DMD

1. Autoencoder reduce dimensionalidad (captura patrones espacio-temporales).
2. DMD se ajusta en el espacio latente tratando dinámica lineal aproximada.
3. Pronóstico: iterar dinámica latente → decodificar → reconstruir precipitación futura.

## MLflow

El script `train_ae_dmd_v2.py` registra:

- Parámetros: `latent_dim`, `epochs`, `batch_size`, `use_lstm`, `use_synthetic`.
- Métricas: MAE, RMSE, NSE, R2, reconstruction_error.
- Artefactos: gráfico comparativo, CSV de predicciones, modelo serializado.

## Extensiones Futuras

- Incorporar metadatos de estaciones (lat/lon reales) sustituyendo grid sintética.
- Añadir ventanas (lookback) para capturar memoria temporal previa al AE.
- Regularización y validación cruzada de `latent_dim` y estructura DMD.
- Integración con ERA5 multi-variable (p.ej. temperatura, presión) para enriquecimiento.

## Convenciones Documentación

- Docstrings en español describen: propósito, entradas, salidas.
- Comentarios explican decisiones no obvias (ej. reshape 6x5, elección de índices).
- Markdown en notebooks añade narrativa del análisis y justificación de pasos.

---
Última actualización: 2025-11-14
