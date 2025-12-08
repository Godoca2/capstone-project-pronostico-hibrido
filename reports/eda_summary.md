# Resumen EDA Espacio-Temporal – Precipitación (Chile)

Fecha: 14-11-2025

## Contexto y Datos

- Fuente base actual: `data/raw/precipitation_data.npy` (fallback temporal).
- Forma del dataset: T x N = 1826 x 30 (días x estaciones/sitios).
- Archivos procesados generados para el EDA:
 - `data/processed/era5_precipitation_chile_full.nc`
 - `data/processed/era5_precipitation_chile_daily.csv`
- Estos archivos permiten ejecutar el notebook `notebooks/01A_Eda_spatiotemporal.ipynb` sin cambios.

## Artefactos del EDA rápido

- Figuras:
 - `reports/figures/mean_time_series.png`: media temporal de precipitación (promedio sobre estaciones).
 - `reports/figures/station_variance.png`: top estaciones por varianza (variabilidad espacial).
- CSVs de apoyo:
 - `reports/eda_time_summary.csv`: media y desviación por día (serie agregada).
 - `reports/eda_station_summary.csv`: media y varianza por estación.

## Hallazgos preliminares

- Variabilidad intranual marcada (ver `mean_time_series.png`), con periodos húmedos y secos.
- Heterogeneidad espacial: algunas estaciones concentran mayor varianza (ver `station_variance.png`).
- Estos patrones justifican: (1) reducción de dimensionalidad (AE), (2) modelado dinámico (DMD/KoVAE) y (3) validación por macrozonas.

> Nota: estos hallazgos se basan en el fallback `.npy`. Al reemplazar por ERA5 2020 real, repetir EDA para confirmar patrones con campo completo.

## Próximos pasos

1. Descargar ERA5 2020 real y consolidar:
 - `python src/utils/download_era5_small.py`
 - `python src/utils/merge_era5_small.py`
2. Re-ejecutar `notebooks/01A_Eda_spatiotemporal.ipynb` con el NetCDF real (`data/processed/era5_precipitation_chile_full.nc`).
3. Exportar productos adicionales del notebook (mapas, correlaciones por macrozona, serie suavizada) a `reports/figures/`.
4. Generar `era5_daily_national_mean.csv` desde el notebook (celda 11) para el pipeline de modelado.

## Reproducibilidad

- Entorno: `conda activate capstone`.
- Dependencias geo/plot: `conda install -y -c conda-forge cartopy proj geos shapely seaborn`.
- Ejecución rápida del EDA de fallback:
 - `python src/quick_eda_npy.py`
- Notebook EDA: `jupyter notebook notebooks/01A_Eda_spatiotemporal.ipynb`.

## Observaciones operativas

- En Windows, para ejecuciones headless usa `$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'` para evitar conflictos de protobuf con matplotlib.
- Si `nbconvert` presenta advertencias de event loop, preferir ejecutar el notebook de forma interactiva.

---
Responsable: César Godoy Delaigue
Proyecto: Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile
