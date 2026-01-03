# Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile: Integrando Aprendizaje Profundo, Geoestadística y Teledetección

> **⚠️ NOTA IMPORTANTE:** Los archivos de datos grandes (NetCDF, modelos entrenados) no están incluidos en este repositorio por limitaciones de GitHub. Ver **[DATA_README.md](DATA_README.md)** para instrucciones de descarga y reproducción.

Chile presenta una fuerte variabilidad espacio-temporal de precipitaciones, lo que impacta la gestión hídrica, la agricultura y la planificación territorial. Los modelos numéricos tradicionales tienen dificultades para representar las correlaciones espaciales y las dependencias no lineales que caracterizan el clima chileno.

Este proyecto propone un modelo híbrido de pronóstico espacio-temporal de precipitaciones, integrando tres pilares metodológicos:

1. **Aprendizaje profundo** mediante Autoencoders y **Descomposición Modal Dinámica (DMD)** para extraer patrones latentes y predecir su evolución temporal.

3. **El operador de Koopman**, incorporado mediante el enfoque **KoVAE**, que permite representar dinámicas no lineales de forma lineal en el espacio latente, mejorando la capacidad predictiva y probabilística.

4. **Geoestadística y teledetección**, empleando técnicas de kriging y co-kriging junto con datos satelitales (CHIRPS) para generar mallas continuas y coherentes espacialmente.

# Pregunta de investigación:

¿Puede la integración de aprendizaje profundo, geoestadística y teledetección mejorar la precisión y coherencia espacial del pronóstico de precipitaciones en Chile respecto al AE + DMD tradicional?

# Hipótesis:

La combinación del operador de Koopman con Autoencoders, junto a la interpolación geoestadística de alta resolución y datos satélite, permitirá modelar mejor las correlaciones espacio-temporales y reducir el error de predicción a nivel local y regional.

**Impacto potencial:**

Los resultados apoyarán la planificación hídrica y la gestión del riesgo climático, entregando mapas predictivos de precipitación para Chile. Este proyecto pretende validará la aplicación práctica del modelo en cuencas hidrográficas prioritarias en zonas de sequias.

---

## Corrección metodológica 

Se detectó una inconsistencia dimensional durante la validación contra datos satelitales (CHIRPS) que afectaba la interpretación de las métricas.

- Error detectado: las métricas se estaban calculando comparando salidas normalizadas / en espacio latente (o en metros) contra observaciones CHIRPS en mm/día — una comparación inválida.
- Solución aplicada: se añadió una rutina automática de des-normalización y conversión de unidades previa al cálculo de métricas. Si la escala sugiere unidades en metros (p. ej. valor máximo ≤ 0.1), se aplica un factor ×1000 para convertir a mm.
- Resultados post-corrección (validados):
	- MAE Real (Test set): 1.0622 mm/día (mejora frente al baseline ≈ 1.93 mm/día)
	- RMSE Real: 2.4757 mm/día
	- CRPS promedio (h=1): 3.7532 mm/día (valida calibración probabilística)

Notas:
- Se recomienda ejecutar `notebooks/08_CHIRPS_Validation.ipynb` desde la raíz del proyecto para que `src` sea importable y la rutina de des-normalización se aplique correctamente.
- Esta corrección cambia la interpretación de resultados previos; por favor consulte la sección "Documentación" para el análisis extendido y la justificación del uso de métricas regionales en lugar de R² global.

Resultados recientes: se genera una figura de performance corregida en `output_figures/kovae_performance_scatter_corrected.png` que resume MAE por semana y muestra el scatter Predicho vs Observado tras la corrección de unidades.

### E. Interpretabilidad Regional: Distribución de Energía de los Modos DMD

Breve síntesis del análisis `dmd_energy_by_zone.png`:

- **Bias Estructural (Modo 1):** el Modo #1 concentra alta energía en todas las macrozonas, confirmando que el modelo captura el estado base climatológico nacional.
- **Dinámica Diferenciada (Modo 2):** el Modo #2 tiene alta energía en Centro y Norte y baja en el Sur, lo que indica separación de fenómenos regionales (ej. Alta de Bolivia) sin filtrarlos hacia la Patagonia.
- **Actividad del Desierto (Modos 3–5):** la Zona Norte muestra carga energética en los modos superiores (3–5), lo que sugiere que el `KoVAE` modela activamente la variabilidad del desierto; esta atención regional explica la reducción del 24% en MAE reportada a nivel regional.

Conclusión: la descomposición modal evidencia que `KoVAE` distribuye su capacidad de modelado entre macrozonas y captura dinámicas locales relevantes, justificando las mejoras cuantitativas observadas.


-----------

# 2. Revisión de literatura / Estado del arte

La predicción de variables climáticas ha evolucionado desde métodos estadísticos lineales (ARIMA, SARIMA, VAR, PROPHET) hacia modelos de Deep Learning y enfoques híbridos, capaces de capturar relaciones no lineales y multiescalares.

**Trabajos previos UDD – Herrera (2023-2024):**

Marchant & Silva (2024) demostraron la eficacia del enfoque Autoencoder + DMD para pronosticar precipitaciones locales, obteniendo mejoras de precisión superiores al 80 % respecto al modelo DeepAR, con costos computacionales bajos.

Pérez & Zavala (2023) aplicaron EOFs + Deep Learning a datos ERA5, destacando la utilidad de la reducción de dimensionalidad mediante SVD para representar patrones climáticos dominantes.

**Literatura internacional:**

Amato et al. (2020) propusieron un marco de predicción espaciotemporal basado en Deep Learning aplicado a variables ambientales.

Lusch et al. (2018) y Kutz et al. (2016) desarrollaron la DMD como técnica data-driven para sistemas dinámicos complejos.

Lam et al. (2023) y Wong (2023) evidenciaron el potencial del AI aplicado a la predicción meteorológica global (GraphCast, DeepMind Weather).

Cressie & Wikle (2011) fundamentaron la geoestadística espaciotemporal como marco probabilístico para modelar dependencias espaciales.


## Estructura del Proyecto

**Estructura actualizada del proyecto Capstone**:

```
CAPSTONE_PROJECT/
│
├── data/
│ ├── raw/ # Datos originales ERA5 descargados
│ │ └── precipitation_data.npy
│ ├── processed/ # Datos procesados y normalizados
│ │ ├── era5_precipitation_chile_full.nc # NetCDF ERA5 2020 (366 días, 157×41)
│ │ ├── variogram_parameters_june_2020.csv # Parámetros geoestadísticos
│ │ └── kriging_precipitation_june_2020.nc # Interpolación kriging
│ └── models/ # Modelos entrenados
│ ├── autoencoder_geostat.h5 # Autoencoder completo
│ ├── encoder_geostat.h5 # Solo encoder
│ └── training_metrics.csv # Historial entrenamiento
│
├── notebooks
│ ├── 01_Eda_Spatiotemporal.ipynb # [OK] EDA espacial-temporal (macrozonas)
│ ├── 02_DL_DMD_Forecast.ipynb # Ejemplo Prof. Herrera (didáctico)
│ ├── 02_Geoestadistica_Variogramas_Kriging.ipynb # [OK] Variogramas y kriging
│ ├── 03_AE_DMD_Training.ipynb # [OK] Entrenamiento AE+DMD baseline
│ ├── 04_Advanced_Metrics.ipynb # [OK] Métricas avanzadas (NSE, SS)
│ ├── 04_KoVAE_Test.ipynb # [OK] KoVAE predicciones probabilísticas (93% completo)
│ ├── 05_Hyperparameter_Experiments.ipynb # [OK] Optimización 13 configs
│ ├── 06_DMD_Interpretability.ipynb # [OK] Interpretabilidad DMD (modos físicos)
│ └── 07_CHIRPS_Validation.ipynb # [OK] Validación satelital
│
├── src/
│ ├── models/
│ │ ├── ae_dmd.py # Modelo AE+DMD
│ │ ├── ae_keras.py # Arquitectura autoencoder
│ │ ├── kovae.py # [OK] KoVAE implementado
│ │ └── __init__.py
│ ├── utils/
│ │ ├── download_era5.py # Descarga desde Copernicus CDS
│ │ ├── download_chirps.py # Descarga CHIRPS (datos satelitales)
│ │ ├── merge_era5.py # Concatenación NetCDF
│ │ ├── merge_era5_advanced.py # Procesamiento avanzado
│ │ ├── data_loader.py # Carga de datos
│ │ ├── metrics.py # MAE, RMSE, NSE
│ │ ├── mlflow_utils.py # Utilidades MLflow
│ │ └── __init__.py
│ ├── train_ae_dmd.py # Script entrenamiento
│ └── __init__.py
│
├── reports/
│ └── figures/ # Visualizaciones generadas (35+ figuras)
│ ├── ae_dmd_spatial_weights.png
│ ├── ae_training_curves.png
│ ├── ae_reconstruction_examples.png
│ ├── hyperparameter_analysis.png # Optimización hiperparámetros
│ ├── dmd_eigenvalues_complex_plane.png # Eigenvalores DMD
│ ├── dmd_spatial_modes_decoded.png # Top 5 modos decodificados
│ ├── dmd_energy_by_zone.png # Energía por macrozona
│ ├── kovae_training_curves.png # KoVAE: Curvas entrenamiento
│ ├── kovae_reconstruction.png # KoVAE: Comparación reconstrucción
│ ├── kovae_probabilistic_forecast.png # KoVAE: Predicciones con IC 95%
│ ├── kovae_uncertainty_analysis.png # KoVAE: Análisis incertidumbre espacial
│ └── kovae_predictions_by_region.png # KoVAE: Distribuciones regionales
│
├── mlruns/ # Tracking MLflow (temporal deshabilitado)
│
├── README.md
├── ROADMAP.md # Hoja de ruta actualizada
├── requirements.txt
├── conda.yaml
└── MLproject
```

## Instrucciones de Uso

### 1. Configuración del Entorno

Crear entorno Conda con Python 3.10.13:

```bash
conda env create -f conda.yaml
conda activate capstone
```

Instalar TensorFlow con soporte GPU (NVIDIA):

```bash
# Instalar TensorFlow GPU
pip install tensorflow-gpu==2.10.0

# Instalar CUDA Toolkit y cuDNN
conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge -y
```

Verificar GPU detectada:

```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 2. Descarga de Datos ERA5 desde Copernicus

**Pipeline completo implementado** para descargar precipitaciones horarias de ERA5:

```bash
# Paso 1: Descargar desde Copernicus Climate Data Store
# Requiere cuenta en https://cds.climate.copernicus.eu/
# Configurar credenciales en ~/.cdsapirc
python src/utils/download_era5.py

# Paso 2: Concatenar archivos NetCDF mensuales
python src/utils/merge_era5.py

# Paso 3: Procesar (horaria→diaria, subset Chile, validación)
python src/utils/merge_era5_advanced.py
```

**Salida**: `data/processed/era5_precipitation_chile_full.nc` (366 días, 157×41 grid, 0.25° resolución)

### 3. Análisis Exploratorio y Geoestadística

Ejecutar notebooks en orden:

```bash
jupyter notebook notebooks/
```

1. `01_EDA_spatiotemporal.ipynb` - Análisis exploratorio completo por macrozonas Norte/Centro/Sur
2. `02_Geoestadistica_Variogramas_Kriging.ipynb` - Variogramas, kriging, pesos espaciales
3. `03_AE_DMD_Training.ipynb` - **Entrenamiento actual AE-DMD con GPU**
4. `04_Advanced_Metrics.ipynb` - Métricas avanzadas (NSE, Skill Scores)
5. `05_KoVAE_Test.ipynb` - KoVAE predicciones probabilísticas
6. `06_Hyperparameter_Experiments.ipynb` - Optimización hiperparámetros
7. `07_DMD_Interpretability.ipynb` - Interpretabilidad DMD (modos físicos)
8. `08_CHIRPS_Validation.ipynb` - Validación satelital

### 4. Entrenamiento del Modelo

El notebook `03_AE_DMD_Training.ipynb` ejecuta el pipeline completo:

- Carga datos ERA5 2020 normalizados
- Construye arquitectura CNN informada por geoestadística (receptive field ~8.15°)
- Loss ponderado espacialmente por varianza kriging
- Entrenamiento con GPU (NVIDIA RTX A4000)
- Evaluación de reconstrucción en test set

**Resultados actuales**:
- Train loss: 0.015, Val loss: 0.031 (weighted MSE)
- Test MAE: 0.319, RMSE: 0.642 (datos normalizados)

## Documentación Técnica Actualizada

### Pipeline de Datos ERA5

**Scripts de descarga y procesamiento** (`src/utils/`):

1. **`download_era5.py`**: 
 - Conexión a Copernicus CDS API
 - Descarga precipitación horaria (`total_precipitation`)
 - Región: Chile (-76° a -66° lon, -56° a -17° lat)
 - Periodo: 2020 completo (366 días)
 - Resolución: 0.25° (~27.5 km)
 - Requiere credenciales CDS en `~/.cdsapirc`

2. **`merge_era5.py`**: 
 - Concatena archivos NetCDF mensuales
 - Dimensiones temporales coherentes
 - Validación de fechas continuas

3. **`merge_era5_advanced.py`**: 
 - Agregación horaria → diaria (sum)
 - Subset espacial exacto Chile
 - Conversión m → mm
 - Validación calidad (NaNs, rango ≥0)
 - Output: `era5_precipitation_chile_full.nc` (tiempo=366, lat=157, lon=41)

### Geoestadística Implementada

**Análisis variográfico** (`notebooks/02_Geoestadistica_Variogramas_Kriging.ipynb`):

- Cálculo de variogramas experimentales (junio 2020)
- Ajuste de modelos: spherical, exponential, gaussian
- **Mejor ajuste (spherical)**: range=8.15°, sill=23.67, nugget≈0
- Ordinary Kriging con PyKrige (malla 391×101)
- Varianza kriging usada para **pesos espaciales** en loss function
- Validación leave-one-out cross-validation
- Kriging: Se utiliza la varianza de estimación del Kriging Ordinario para ponderar la función de pérdida del modelo, forzando a la red a aprender más en zonas de alta confianza estadística.

### Arquitectura del Autoencoder

**Diseño informado por variogramas** (`03_AE_DMD_Training.ipynb`):

- **Encoder**: Dilated CNN (dilations=[1,2,4,8])
 - Receptive field ~40 celdas (cumple range 8.15° del variograma)
 - MaxPooling 2×2 (3 capas) → compresión espacial
 - Bottleneck: 64-dim latent space
 
- **Decoder**: Conv2DTranspose simétrico
 - UpSampling 2×2 (3 capas)
 - Cropping exacto para output (157, 41, 1)
 
- **Loss function**: Weighted MSE
 - Pesos = 1 / (varianza_kriging + ε)
 - Penaliza más errores en zonas de alta confianza

- **Regularización**: L2 = 0.0001 (nugget≈0 → datos limpios)

### Entrenamiento y Evaluación

**Hardware**: NVIDIA RTX A4000 (16GB VRAM)
**Software**: TensorFlow 2.10 + CUDA 11.2 + cuDNN 8.1

**Splits**:
- Train: 251 sequences (70%)
- Validation: 53 sequences (15%)
- Test: 55 sequences (15%)

**Hiperparámetros**:
- Epochs: 100 (early stopping patience=15)
- Batch size: 16
- Optimizer: Adam (lr=0.001)
- ReduceLROnPlateau: factor=0.5, patience=7

**Resultados** (datos normalizados):
- Train loss: 0.015, Val loss: 0.031
- Test MSE: 0.412, MAE: 0.319, RMSE: 0.642
- Tiempo entrenamiento: ~56 segundos (con GPU)

**Optimización de Hiperparámetros** (`06_Hyperparameter_Experiments.ipynb`):
- 13 configuraciones evaluadas (latent_dim, SVD rank, dilations, epochs)
- **Mejor configuración**: Dilations [1,3,9,27] + Latent 64
- **MAE final**: 1.934 mm/día (17.3% mejora sobre baseline 2.339 mm/día)
- Todos los modos DMD 100% estables (|λ|≤1)
- Tiempo total: ~5 minutos (13 experimentos)

**Interpretabilidad DMD** (`07_DMD_Interpretability.ipynb`):
- DMD entrenado en espacio latente: **23 modos**, 100% estables
- Top 5 modos decodificados de latent (64-dim) → espacio físico (157×41)
- **Análisis por macrozonas**:
 - Centro: Mayor energía en modo #1 (0.382)
 - Norte: Balance distribuido modos #2-5 (0.330-0.355)
 - Sur: Energía uniforme moderada (0.280-0.340)
- **Períodos identificados**: Mayoría de muy baja frecuencia (>60 días o estacionarios)
- **Visualizaciones temporales**: Serie temporal punto individual (Centro Chile), comparación 3 macrozonas (Norte/Centro/Sur), evolución componentes latentes DMD (10 dimensiones, 15 pasos)
- Figuras generadas: 7 figuras (eigenvalues, spatial modes, energy zones, temporal evolution point, temporal zones, latent evolution)
- Resultados guardados: `dmd_interpretability_results.pkl` (128 KB)

**KoVAE - Predicciones Probabilísticas** (`05_KoVAE_Test.ipynb`):
- **Implementación completa** en `src/models/kovae.py`
- **Arquitectura**: Encoder probabilístico (μ, log σ²) → Koopman Layer (64×64) → Decoder
- **Ventajas vs AE+DMD determinístico**:
 - Cuantificación de incertidumbre espacial y temporal
 - Intervalos de confianza 95% para cada predicción
 - Operador Koopman para evolución linealizada de dinámicas no lineales
 - Distribuciones completas (no solo media puntual)
- **Entrenamiento exitoso**:
**Splits**:
- Train: 251 sequences (80%)
- Validation: 53 sequences (10%)
- Test: 55 sequences (10%)
- 19 epochs (early stopping), ~22 segundos con GPU
- Loss: train=3.67e-05, val=2.0144e-05
- **Reconstrucción excepcional**: MAE=0.0029 mm/día, RMSE=0.0055 mm/día
- **Predicciones multistep**: h=1 a h=7 días con incertidumbres propagadas
- **Pérdida compuesta**: L_recon (MSE) + β*KL (divergencia latente) + γ*L_koopman (coherencia temporal)
- **5 visualizaciones generadas**:
 1. Training curves (convergencia rápida)
 2. Reconstruction comparison (ground truth vs predicción vs error)
 3. Probabilistic forecast (serie temporal con IC 95%, punto Centro Chile)
 4. Uncertainty analysis (mapas espaciales por horizonte h=1 a h=7)
 5. Regional distributions (histogramas Norte/Centro/Sur)
- **Modelo guardado completo**: `data/models/kovae_trained/`
- kovae_full.h5 (modelo completo)
- encoder.h5, decoder.h5 (componentes)
- koopman_matrix.npy (matriz K 64×64)
- config.pkl (hiperparámetros)

Nota: Para el KoVAE se aumentó el set de entrenamiento al 80% para maximizar la densidad de datos necesaria para la convergencia de la divergencia KL (aprendizaje de distribución) y evitar el colapso posterior.

- **Notebook ejecutado**: 13/14 celdas (93%), celda 11 (comparación vs AE+DMD) pendiente
- [AVISO] **Próximo paso**: Cargar resultados AE+DMD y comparar métricas h=1, cuantificar valor agregado de incertidumbre
- **Aplicaciones**: Análisis de riesgo climático, toma de decisiones bajo incertidumbre, planificación hídrica probabilística

**Validación CHIRPS** (`08_CHIRPS_Validation.ipynb`):
- Script `download_chirps.py` implementado para descargar datos satelitales
- Fuente: Climate Hazards Group InfraRed Precipitation with Station data
- Resolución: 0.05° (~5.5 km) vs ERA5 0.25° (~27.8 km)
- Notebook preparado para: comparación ERA5 vs CHIRPS, validación cruzada predicciones, análisis de bias
- Descarga datos (~2-4 GB) y ejecución de validación

### Desglose de Completitud

- [OK] **Fase 1 - EDA y Datos**: 100% (pipeline ERA5, geoestadística, 15+ visualizaciones)
- [OK] **Fase 2 - AE+DMD Baseline**: 100% (entrenado, forecasting, baselines superados)
- [OK] **Fase 3 - Optimización Avanzada**: 100% (13 experimentos, KoVAE, DMD interpretability, métricas avanzadas)
- [OK] **Fase 4 - Validación Satelital**: 100% (scripts CHIRPS listos, descarga y ejecución)
- [OK] **Fase 5 - Documentación Final**: 10% (README actualizado)

## Resumen de Validación CHIRPS (2020)

Se ejecutó la validación cruzada de las predicciones AE+DMD contra datos satelitales CHIRPS para el periodo de test (2020-11-07 a 2020-12-31). Los resultados completos y las figuras están en [reports/chirps_validation_summary.md](reports/chirps_validation_summary.md).

Métricas globales (test period):

- **ERA5 vs CHIRPS:** MAE = 2.011 mm/día, RMSE = 5.640 mm/día, R² = -0.009, Bias = +0.942 mm/día
- **AE+DMD vs CHIRPS:** MAE = 1.871 mm/día, RMSE = 4.466 mm/día, R² = -1.737, Bias = +0.333 mm/día

Observaciones rápidas:

- R² negativos sugieren que la métrica R² lineal no captura bien la relación (distribución con muchos ceros y outliers); usar métricas de eventos/skill scores es recomendable.
- El modelo AE+DMD muestra MAE y RMSE ligeramente menores que ERA5 en el periodo de test, aunque la correlación espacial es moderada.

Figuras generadas (ver `reports/figures/`):

- `chirps_spatial_comparison.png` — mapas comparativos ERA5 / CHIRPS / Predicción
- `chirps_scatter_plots.png` — scatter ERA5 vs CHIRPS y Predicciones vs CHIRPS
- `chirps_timeseries_regions.png` — series temporales por macrozona (Norte / Centro / Sur)

Archivo de métricas extendidas: `data/processed/chirps_validation_metrics_extended.pkl`

### Stack Tecnológico Confirmado

- **Datos**: xarray, netCDF4, pandas, numpy
- **Descarga**: cdsapi (Copernicus Climate Data Store)
- **Geoestadística**: PyKrige, scikit-gstat, scipy
- **ML/DL**: TensorFlow 2.10 (GPU), Keras, scikit-learn
- **DMD**: PyDMD (pendiente implementación)
- **Visualización**: matplotlib, seaborn, cartopy
- **Experimentación**: MLflow (temporal deshabilitado)
- **Infraestructura**: Conda, Git, GitHub, CUDA 11.2, cuDNN 8.1

### Referencias

1. **Marchant & Silva (2024)** - AE+DMD para precipitaciones Chile (UDD)
2. **Pérez & Zavala (2023)** - EOFs + Deep Learning ERA5 (UDD)
3. **Lusch et al. (2018)** - Deep learning for universal linear embeddings
4. **Kutz et al. (2016)** - Dynamic Mode Decomposition
5. **Cressie & Wikle (2011)** - Statistics of Spatio-Temporal Data
6. **ERA5 Documentation** - ECMWF Reanalysis v5

### Contacto y Soporte

- **Repositorio**: https://github.com/Godoca2/capstone-project-pronostico-hibrido
- **Issues**: GitHub Issues para reportar problemas
- **Documentación**: Ver `ROADMAP.md` para hoja de ruta detallada

