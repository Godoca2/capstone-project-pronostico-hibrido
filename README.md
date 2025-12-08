# Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile: Integrando Aprendizaje Profundo, Geoestadística y Teledetección

> **⚠️ NOTA IMPORTANTE:** Los archivos de datos grandes (NetCDF, modelos entrenados) no están incluidos en este repositorio por limitaciones de GitHub. Ver **[DATA_README.md](DATA_README.md)** para instrucciones de descarga y reproducción.

Chile presenta una fuerte variabilidad espacio-temporal de precipitaciones, lo que impacta la gestión hídrica, la agricultura y la planificación territorial. Los modelos numéricos tradicionales tienen dificultades para representar las correlaciones espaciales y las dependencias no lineales que caracterizan el clima chileno.

Este proyecto propone un modelo híbrido de pronóstico espacio-temporal de precipitaciones, integrando tres pilares metodológicos:

1. **Aprendizaje profundo** mediante Autoencoders y **Descomposición Modal Dinámica (DMD)** para extraer patrones latentes y predecir su evolución temporal.

3. **El operador de Koopman**, incorporado mediante el enfoque **KoVAE**, que permite representar dinámicas no lineales de forma lineal en el espacio latente, mejorando la capacidad predictiva y probabilística.

4. **Geoestadística y teledetección**, empleando técnicas de kriging y co-kriging junto con datos satelitales (CHIRPS, GPM y MODIS) para generar mallas continuas y coherentes espacialmente.

# Pregunta de investigación:

¿Puede la integración de aprendizaje profundo, geoestadística y teledetección mejorar la precisión y coherencia espacial del pronóstico de precipitaciones en Chile respecto al AE + DMD tradicional?

# Hipótesis:

La combinación del operador de Koopman con Autoencoders, junto a la interpolación geoestadística de alta resolución y datos satélite, permitirá modelar mejor las correlaciones espacio-temporales y reducir el error de predicción a nivel local y regional.

**Impacto potencial:**

Los resultados apoyarán la planificación hídrica y la gestión del riesgo climático, entregando mapas predictivos de precipitación para Chile. Este proyecto pretende validará la aplicación práctica del modelo en cuencas hidrográficas prioritarias en zonas de sequias.

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

---

## Glosario de Conceptos Técnicos

### **Autoencoder (AE)**
Red neuronal no supervisada que comprime datos (encoder) y los reconstruye (decoder). Usado para capturar patrones espaciales de precipitación en representación compacta.

### **Espacio Latente**
Representación de menor dimensión (ej: 64-dim) de datos originales (6437 celdas). Reduce complejidad preservando información esencial.

### **DMD (Descomposición Modal Dinámica)**
Técnica data-driven que descompone sistemas dinámicos en modos espacio-temporales coherentes. Extrae patrones + frecuencias para pronósticos.

### **KoVAE (Koopman Variational Autoencoder)**
Extensión probabilística del Autoencoder que usa el Operador de Koopman para representar dinámicas no lineales como lineales. Incluye incertidumbre.

### **Variograma**
Función que cuantifica correlación espacial vs distancia. Parámetros: nugget (error), sill (varianza máx), range (alcance correlación).

### **Kriging**
Interpolación geoestadística óptima que genera campos continuos + varianza de estimación a partir de observaciones puntuales.

### **Dilated Convolutions**
Convoluciones con "huecos" que expanden campo receptivo sin aumentar parámetros. Captura contexto multi-escala.

### **Métricas**
- **MAE**: Error promedio absoluto (mm/día)
- **RMSE**: Raíz error cuadrático medio
- **NSE**: Eficiencia Nash-Sutcliffe (hidrología)
- **Skill Score**: Mejora % vs baseline

---

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
├── notebooks/
│ ├── 01_EDA.ipynb # [OK] EDA básico
│ ├── 01A_Eda_spatiotemporal.ipynb # [OK] EDA espacial-temporal (macrozonas)
│ ├── 02_DL_DMD_Forecast.ipynb # Ejemplo Prof. Herrera (didáctico)
│ ├── 02_Geoestadistica_Variogramas_Kriging.ipynb # [OK] Variogramas y kriging
│ ├── 03_AE_DMD_Training.ipynb # [OK] Entrenamiento AE+DMD baseline
│ ├── 04_Advanced_Metrics.ipynb # [OK] Métricas avanzadas (NSE, SS)
│ ├── 04_KoVAE_Test.ipynb # [OK] KoVAE predicciones probabilísticas (93% completo)
│ ├── 05_Hyperparameter_Experiments.ipynb # [OK] Optimización 13 configs
│ ├── 06_DMD_Interpretability.ipynb # [OK] Interpretabilidad DMD (modos físicos)
│ └── 07_CHIRPS_Validation.ipynb # [En Espera] Validación satelital (preparado)
│
├── src/
│ ├── models/
│ │ ├── ae_dmd.py # Modelo AE+DMD
│ │ ├── ae_keras.py # Arquitectura autoencoder
│ │ ├── kovae.py # [OK] KoVAE implementado (400+ líneas)
│ │ └── __init__.py
│ ├── utils/
│ │ ├── download_era5.py # Descarga desde Copernicus CDS
│ │ ├── download_chirps.py # [OK] Descarga CHIRPS (datos satelitales)
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
│ ├── kovae_training_curves.png # [OK] KoVAE: Curvas entrenamiento
│ ├── kovae_reconstruction.png # [OK] KoVAE: Comparación reconstrucción
│ ├── kovae_probabilistic_forecast.png # [OK] KoVAE: Predicciones con IC 95%
│ ├── kovae_uncertainty_analysis.png # [OK] KoVAE: Análisis incertidumbre espacial
│ └── kovae_predictions_by_region.png # [OK] KoVAE: Distribuciones regionales
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
- Construye arquitectura CNN informada por geoestadística (receptive field ~8.23°)
- Loss ponderado espacialmente por varianza kriging
- Entrenamiento con GPU (NVIDIA RTX A4000): ~56 segundos
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
- **Mejor ajuste (spherical)**: range=8.23°, sill=23.45, nugget≈0
- Ordinary Kriging con PyKrige (malla 391×101)
- Varianza kriging usada para **pesos espaciales** en loss function
- Validación leave-one-out cross-validation

### Arquitectura del Autoencoder

**Diseño informado por variogramas** (`03_AE_DMD_Training.ipynb`):

- **Encoder**: Dilated CNN (dilations=[1,2,4,8])
 - Receptive field ~40 celdas (cumple range 8.23° del variograma)
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

**Optimización de Hiperparámetros** (`05_Hyperparameter_Experiments.ipynb`):
- 13 configuraciones evaluadas (latent_dim, SVD rank, dilations, epochs)
- **Mejor configuración**: Dilations [1,3,9,27] + Latent 64
- **MAE final**: 1.934 mm/día (17.3% mejora sobre baseline 2.339 mm/día)
- Todos los modos DMD 100% estables (|λ|≤1)
- Tiempo total: ~5 minutos (13 experimentos)

**Interpretabilidad DMD** (`06_DMD_Interpretability.ipynb`):
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

**KoVAE - Predicciones Probabilísticas** (`04_KoVAE_Test.ipynb`):
- **Implementación completa** en `src/models/kovae.py` (407 líneas)
- **Arquitectura**: Encoder probabilístico (μ, log σ²) → Koopman Layer (64×64) → Decoder
- **Ventajas vs AE+DMD determinístico**:
 - Cuantificación de incertidumbre espacial y temporal
 - Intervalos de confianza 95% para cada predicción
 - Operador Koopman para evolución linealizada de dinámicas no lineales
 - Distribuciones completas (no solo media puntual)
- **Entrenamiento exitoso**:
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
- **Notebook ejecutado**: 13/14 celdas (93%), celda 11 (comparación vs AE+DMD) pendiente
- [AVISO] **Próximo paso**: Cargar resultados AE+DMD y comparar métricas h=1, cuantificar valor agregado de incertidumbre
- **Aplicaciones**: Análisis de riesgo climático, toma de decisiones bajo incertidumbre, planificación hídrica probabilística

**Validación CHIRPS** (`07_CHIRPS_Validation.ipynb`):
- Script `download_chirps.py` implementado para descargar datos satelitales
- Fuente: Climate Hazards Group InfraRed Precipitation with Station data
- Resolución: 0.05° (~5.5 km) vs ERA5 0.25° (~27.8 km)
- Notebook preparado para: comparación ERA5 vs CHIRPS, validación cruzada predicciones, análisis de bias
- [AVISO] **Pendiente**: Descarga datos (~2-4 GB) y ejecución de validación

### Próximos Pasos (Opcionales)

Ver `ROADMAP.md` para tareas pendientes:

1. [OK] ~~**DMD en espacio latente**~~ - Completado (23 modos, 100% estables)
2. [OK] ~~**Desnormalización**~~ - Métricas en mm/día reales
3. [OK] ~~**Análisis por macrozonas**~~ - Norte/Centro/Sur evaluados
4. [OK] ~~**Baselines**~~ - Persistencia y climatología implementados
5. [OK] ~~**Optimización hiperparámetros**~~ - 13 configs, MAE 1.934 mm/día
6. [OK] ~~**Interpretabilidad DMD**~~ - Modos decodificados a espacio físico
7. **Validación CHIRPS** - Comparar con datos satelitales (opcional)
8. **KoVAE** - Implementar operador de Koopman variacional (opcional)
9. **Resolver MLflow** - Conflicto protobuf (MLflow 3.6 vs TF 2.10)

---

## Referencias y Metadatos

**Última actualización**: 19 noviembre 2025 
**Responsable**: César Godoy Delaigue 
**Fase actual**: Fase 3 - Optimización Avanzada (**Completada 100%**) → Iniciando Fase 4 
**Notebooks ejecutados**: 7/8 principales (87.5%), todos funcionales 
**Progreso global**: **65%** (3/5 fases completas)

### Desglose de Completitud

- [OK] **Fase 1 - EDA y Datos**: 100% (pipeline ERA5, geoestadística, 15+ visualizaciones)
- [OK] **Fase 2 - AE+DMD Baseline**: 100% (entrenado, forecasting, baselines superados)
- [OK] **Fase 3 - Optimización Avanzada**: 100% (13 experimentos, KoVAE, DMD interpretability, métricas avanzadas)
- [En Progreso] **Fase 4 - Validación Satelital**: 15% (scripts CHIRPS listos, descarga y ejecución pendiente)
- [En Espera] **Fase 5 - Documentación Final**: 10% (README actualizado, paper draft pendiente)

### Stack Tecnológico Confirmado

- **Datos**: xarray, netCDF4, pandas, numpy
- **Descarga**: cdsapi (Copernicus Climate Data Store)
- **Geoestadística**: PyKrige, scikit-gstat, scipy
- **ML/DL**: TensorFlow 2.10 (GPU), Keras, scikit-learn
- **DMD**: PyDMD (pendiente implementación)
- **Visualización**: matplotlib, seaborn, cartopy
- **Experimentación**: MLflow (temporal deshabilitado)
- **Infraestructura**: Conda, Git, GitHub, CUDA 11.2, cuDNN 8.1

### Referencias Clave

1. **Marchant & Silva (2024)** - AE+DMD para precipitaciones Chile (UDD)
2. **Pérez & Zavala (2023)** - EOFs + Deep Learning ERA5 (UDD)
3. **Lusch et al. (2018)** - Deep learning for universal linear embeddings
4. **Kutz et al. (2016)** - Dynamic Mode Decomposition
5. **Cressie & Wikle (2011)** - Statistics of Spatio-Temporal Data
6. **ERA5 Documentation** - ECMWF Reanalysis v5

### Contacto y Soporte

- **Repositorio**: https://github.com/Godoca2/Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile
- **Issues**: GitHub Issues para reportar problemas
- **Documentación**: Ver `ROADMAP.md` para hoja de ruta detallada

