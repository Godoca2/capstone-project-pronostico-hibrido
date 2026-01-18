# Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile
## Integrando Aprendizaje Profundo, Geoestadística y Teledetección

**Autor:** César Godoy Delaigue  
**Institución:** Universidad del Desarrollo (UDD)  
**Programa:** Magíster en Data Science  
**Fecha:** Enero 2026

---

## Índice

1. [Resumen](#1-resumen)
2. [Problema de Investigación](#2-problema-de-investigación)
3. [Marco Teórico](#3-marco-teórico)
4. [Metodología](#4-metodología)
5. [Datos](#5-datos)
6. [Resultados](#6-resultados)
7. [Estructura del Proyecto](#7-estructura-del-proyecto)
8. [Instalación y Reproducción](#8-instalación-y-reproducción)
9. [Referencias](#9-referencias)
10. [Contacto](#10-contacto)

---

## 1. Resumen

Este proyecto desarrolla un sistema híbrido de pronóstico de precipitaciones para Chile continental, combinando **Aprendizaje Profundo** (Autoencoders Convolucionales), **Análisis de Sistemas Dinámicos** (DMD - Dynamic Mode Decomposition) y **Geoestadística** (Kriging y Variogramas).

<!-- ### Resultados Principales

| Métrica | Valor |
|---------|-------|
| **MAE** | 1.763 mm/día (±0.04) |
| **Mejora vs Persistence** | +7.1% |
| **Mejora vs Climatología** | +12.9% |
| **Compresión espacial** | 100.3× (6,437 → 64 dim) |
| **Validación CHIRPS** | +7% mejor que ERA5 crudo | -->

> **⚠️ NOTA:** Los archivos de datos grandes (NetCDF, modelos entrenados) no están incluidos por limitaciones de GitHub. Ver **[DATA_README.md](DATA_README.md)** para instrucciones de descarga.

---

## 2. Problema de Investigación

### 2.1 Contexto

Chile presenta una fuerte variabilidad espacio-temporal de precipitaciones debido a su extensión latitudinal (17°S - 56°S) y la presencia de la Cordillera de los Andes. Esta variabilidad impacta directamente en:

- Gestión de recursos hídricos
- Planificación agrícola
- Gestión del riesgo climático
- Ordenamiento territorial

Los modelos numéricos tradicionales tienen dificultades para representar las correlaciones espaciales y las dependencias no lineales que caracterizan el clima chileno.

### 2.2 Pregunta de Investigación

> *¿Puede la integración de aprendizaje profundo, geoestadística y teledetección mejorar la precisión y coherencia espacial del pronóstico de precipitaciones en Chile respecto al enfoque AE+DMD tradicional?*

### 2.3 Hipótesis

La combinación del operador de Koopman con Autoencoders, junto a la interpolación geoestadística de alta resolución y datos satelitales, permitirá modelar mejor las correlaciones espacio-temporales y reducir el error de predicción a nivel local y regional.

---

## 3. Marco Teórico

### 3.1 Estado del Arte

La predicción de variables climáticas ha evolucionado desde métodos estadísticos lineales (ARIMA, SARIMA) hacia modelos de Deep Learning y enfoques híbridos.

**Trabajos previos UDD:**
- Marchant & Silva (2024): AE+DMD para precipitaciones locales, mejoras >80% vs DeepAR
- Pérez & Zavala (2023): EOFs + Deep Learning aplicado a datos ERA5

**Literatura internacional:**
- Lusch et al. (2018): Deep learning para embeddings lineales universales
- Kutz et al. (2016): Dynamic Mode Decomposition para sistemas dinámicos
- Lam et al. (2023): GraphCast (DeepMind) para predicción meteorológica global
- Cressie & Wikle (2011): Geoestadística espaciotemporal

### 3.2 Pilares Metodológicos

1. **Autoencoders Convolucionales:** Compresión de campos espaciales preservando estructura
2. **DMD (Dynamic Mode Decomposition):** Extracción de modos dinámicos y proyección temporal
3. **Operador de Koopman (KoVAE):** Linealización de dinámicas no lineales en espacio latente
4. **Geoestadística:** Variogramas y Kriging para coherencia espacial

---

## 4. Metodología

### 4.1 Fase 1: Preparación de Datos (Notebooks 01-02)

**Objetivo:** Preparar y caracterizar los datos antes del modelado.

```
┌─────────────────────────────────────┐
│         INGESTA DE DATOS            │
│   ERA5 Reanalysis (0.25°, horario)  │
│   366 días 2020, Grid 157×41        │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│       PREPROCESAMIENTO              │
│   - Agregación horaria → diaria     │
│   - Conversión m → mm/día           │
│   - Validación (NaNs, outliers)     │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│    ANÁLISIS EXPLORATORIO (EDA)      │
│   - Estadísticas por macrozona      │
│   - Patrones estacionales           │
│   - Detección de extremos           │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│       GEOESTADÍSTICA                │
│   - Variogramas experimentales      │
│   - Modelo esférico (Range=8.15°)   │
│   - Kriging Ordinario               │
│   - Cálculo varianza kriging        │
└─────────────────────────────────────┘
```

**Outputs:** 
- Dataset procesado: `era5_precipitation_chile_full.nc`
- Parámetros variograma: Range=8.15°, Sill=23.45, Nugget=0.0
- Pesos espaciales para loss function

---

### 4.2 Fase 2A: Pipeline Base AE+DMD (Notebook 03)

**Objetivo:** Línea base determinista con proyección temporal DMD.

```
┌─────────────────────────────────────┐
│     DATOS PREPROCESADOS             │
│   Split: 70% / 15% / 15%            │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      STANDARD SCALER                │
│   Media=0, Varianza=1               │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│   AUTOENCODER DETERMINÍSTICO        │
│   - Encoder: Conv2D dilatadas       │
│   - Latent: 64 dim                  │
│   - Decoder: Conv2DTranspose        │
│   - Loss: Weighted MSE (Kriging)    │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         ESPACIO LATENTE             │
│   Embeddings 64-dim                 │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│              DMD                    │
│   - 23 modos estables               │
│   - Proyección: z_{t+h} = A^h·z_t   │
│   - Horizontes: h = 1, 3, 7 días    │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      PREDICCIÓN PROMEDIO            │
│   Una única predicción determinista │
└─────────────────────────────────────┘
```

**Validación interna:** MAE/RMSE vs Test Set (ERA5)  
**Resultado:** MAE = 1.934 mm/día (baseline)

---

### 4.3 Fase 2B: Pipeline KoVAE Physics-Informed (Notebook 05)

**Objetivo:** Modelo probabilístico con operador de Koopman para cuantificar incertidumbre.

```
┌─────────────────────────────────────┐
│     DATOS PREPROCESADOS             │
│   Split: 80% / 10% / 10%            │
│   (Mayor densidad para VAE)         │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      MINMAX SCALER [0,1]            │
│   (Evita KL collapse)               │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│     ENCODER PROBABILÍSTICO          │
│   - Conv2D → Flatten → Dense        │
│   - Output: μ (media), σ (varianza) │
│   - Sampling: z = μ + σ·ε           │
│   - Latent: 128 dim                 │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│     OPERADOR DE KOOPMAN (K)         │
│   - Matriz lineal 128×128           │
│   - Evolución: z_{t+1} = K·z_t      │
│   - Proyección multi-step: K^h      │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│           DECODER                   │
│   - Dense → Reshape → Conv2DT       │
│   - Reconstrucción espacial (157×41)│
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│    PREDICCIÓN PROBABILÍSTICA        │
│   - 30 muestras Monte Carlo         │
│   - Intervalos de confianza 95%     │
│   - Mapas de incertidumbre          │
└─────────────────────────────────────┘
```

**Pérdida compuesta:** L = L_recon + β·KL + γ·L_koopman  
- β = 0.005 (evita posterior collapse)
- γ = 0.5 (coherencia temporal)

**Resultado:** MAE = 1.070 mm/día (+44.7% vs baseline)

---

### 4.4 Fase 3: Validación Unificada (Notebook 08)

**Objetivo:** Comparar ambos modelos contra datos satelitales independientes.

```
┌─────────────────────────────────────┐
│       PREDICCIONES                  │
│   AE+DMD (determinista)             │
│         vs                          │
│   KoVAE (probabilística)            │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      JUEZ INDEPENDIENTE             │
│   CHIRPS Satelital (0.05°)          │
│   Resolución 5× mejor que ERA5      │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      MÉTRICAS COMPARATIVAS          │
│   - MAE por macrozona               │
│   - Bias regional                   │
│   - CSI (eventos extremos)          │
└─────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────────────┐
│         RESULTADO FINAL                        │
│  Mejor Desenpeño KoVAE mejor: MAE 1.07 vs 1.87 │
│  Norte: +24.3% mejor                           │
│  Sur: +28.3% mejor                             │
└────────────────────────────────────────────────┘
```

**Conclusión clave:** KoVAE corrige "lluvia fantasma" en el Norte (zona árida) gracias al operador de Koopman.

---

## 5. Datos

### 5.1 Fuentes de Datos

| Dataset | Resolución | Cobertura | Uso |
|---------|------------|-----------|-----|
| **ERA5** | 0.25° (~28 km) | 2020 (366 días) | Entrenamiento |
| **CHIRPS** | 0.05° (~5.5 km) | 2020 | Validación |

### 5.2 Región de Estudio

- **Latitud:** -56° a -17.5° (Chile continental)
- **Longitud:** -76° a -66°
- **Grid:** 157 × 41 celdas (6,437 píxeles)

### 5.3 Splits de Datos

**Modelo AE+DMD (determinístico):**

| Conjunto | Proporción | Secuencias |
|----------|------------|------------|
| Train | 70% | 251 |
| Validation | 15% | 53 |
| Test | 15% | 55 |

**Modelo KoVAE (probabilístico):**

| Conjunto | Proporción | Secuencias | Justificación |
|----------|------------|------------|---------------|
| Train | 80% | 286 | Mayor densidad para convergencia KL |
| Validation | 10% | 35 | Suficiente para early stopping |
| Test | 10% | 35 | Evaluación final independiente |

> **¿Por qué splits diferentes?** Los VAEs aprenden distribuciones de probabilidad, no solo mappings determinísticos. La divergencia KL en la pérdida requiere suficientes ejemplos para que el encoder aprenda a generar distribuciones latentes útiles sin colapsar a la media.

---

## 6. Resultados

### 6.1 Modelo Baseline (AE+DMD - Notebook 04)

**Validación interna:** Test Set ERA5 (15% de los datos, 55 secuencias)

| Métrica | Valor |
|---------|-------|
| **MAE** | 1.934 mm/día |
| **RMSE** | 4.305 mm/día |
| **vs Persistence** | +7.1% |
| **vs Climatology** | +12.9% |

**Conclusión:** Establece línea base determinista. El modelo supera benchmarks triviales, pero sin cuantificación de incertidumbre.

---

### 6.2 Modelo Propuesto (KoVAE - Notebook 05)

**Validación interna:** Test Set ERA5 (10% de los datos, 35 secuencias)

| Métrica | Valor | Intervalo 95% |
|---------|-------|---------------|
| **MAE Real** | **1.068 mm/día** | [0.982, 1.158] |
| **RMSE Real** | **2.457 mm/día** | [2.203, 2.711] |
| **CRPS** | 3.805 mm/día | [3.512, 4.098] |

> **Nota crítica:** Las métricas originales estaban en escala normalizada MinMax [0,1]. Se aplicó desnormalización: cuando max(predicción) ≤ 0.1, el sistema detecta automáticamente unidades en metros y convierte a mm/día (×1000). La métrica de reconstrucción interna del VAE (0.0029) **NO** es la precisión de pronóstico.

**Métricas Probabilísticas:**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **CRPS** | 3.805 mm/día | Calidad de la distribución predicha |
| **IC 95%** | P10-P90 | Intervalos de confianza calibrados |
| **Calibración** | ✓ Validada | Observaciones caen dentro del IC |

**Mejora KoVAE vs AE+DMD:** +44.7% en MAE (1.070 vs 1.934 mm/día)

---

### 6.3 Benchmark Comparison (Notebook 09)

**Comparacion de cuatro modelos de pronostico a 1 dia (h=1):**

| Modelo | MAE (mm/dia) | RMSE (mm/dia) | POD >1mm | Observacion |
|--------|--------------|---------------|----------|-------------|
| **ERA5 (Persistence)** | 1.86 | 4.12 | 0.678 | Baseline fisico |
| **ConvLSTM** | 1.45 | 3.89 | 0.312 | Mode collapse |
| **AE+DMD** | **1.72** | 3.95 | 0.746 | Mejor balance MAE-POD |
| **KoVAE** | 2.41 | 4.87 | **0.815** | Mejor deteccion |

**Hallazgos clave:**
- ConvLSTM sufre "mode collapse" en datos de precipitacion (~85% ceros)
- Las arquitecturas encoder-decoder superan a ConvLSTM end-to-end
- Trade-off deteccion vs precision: KoVAE detecta mas eventos pero con mayor MAE

---

### 6.4 Validacion Unificada CHIRPS (Notebook 08)

**Juez Independiente:** CHIRPS Satelital (0.05°, resolucion 5x mejor que ERA5)

**Comparación AE+DMD vs KoVAE vs ERA5:**

| Modelo | MAE vs CHIRPS (mm/día) | Mejora vs ERA5 |
|--------|------------------------|----------------|
| **ERA5 Reanalysis** | 2.0112 | - (baseline) |
| **AE+DMD** | 1.8708 | +7.0% |
| **KoVAE (mejor)** | **1.0638** | **+47.1%** |

**Desempeño Regional (KoVAE vs ERA5):**

| Macrozona | Latitudes | ERA5 MAE | KoVAE MAE | Mejora |
|-----------|-----------|----------|-----------|--------|
| **Norte** | -17° a -30° (Atacama) | 1.45 mm/día | 1.10 mm/día | **+24.3%** |
| Centro | -30° a -40° | 1.92 mm/día | 1.48 mm/día | +22.9% |
| **Sur** | -40° a -56° (Patagonia) | 2.31 mm/día | 1.66 mm/día | **+28.3%** |

> **Hallazgo clave:** El operador de Koopman corrige la "lluvia fantasma" de ERA5 en el Desierto de Atacama (Norte). AE+DMD, sin la capa Koopman, no logra esta corrección física.

---

### 6.5 Analisis DMD (Notebook 07)

**Modos estables identificados:**
- Total: **23 modos** (100% estables)
- Cobertura temporal: 365 días

**Interpretación física:**

| Modo | Descripción |
|------|-------------|
| #1 | Estado base climatológico nacional |
| #2 | Separación fenómenos regionales (Alta de Bolivia) |
| #3-5 | Variabilidad regional diferenciada (Norte-Centro-Sur) |

---

## 7. Estructura del Proyecto

```
CAPSTONE_PROJECT/
├── data/
│   ├── raw/                    # Datos originales ERA5
│   ├── processed/              # Datos procesados
│   │   ├── era5_precipitation_chile_full.nc
│   │   └── variogram_parameters_june_2020.csv
│   └── models/                 # Modelos entrenados
│       ├── autoencoder_geostat.h5
│       ├── encoder_geostat.h5
│       ├── kovae_trained/
│       └── training_metrics.csv
│
├── notebooks/                  # Análisis y experimentos
│   ├── 01_EDA_Spatiotemporal.ipynb
│   ├── 02_Geoestadistica_Variogramas_Kriging.ipynb
│   ├── 03_AE_DMD_Training.ipynb
│   ├── 04_Advanced_Metrics.ipynb
│   ├── 05_KoVAE_Test.ipynb
│   ├── 06_Hyperparameter_Experiments.ipynb
│   ├── 07_DMD_Interpretability.ipynb
│   ├── 08_CHIRPS_Validation.ipynb
│   └── 09_Benchmark_Comparison.ipynb
│
├── src/                        # Código fuente
│   ├── models/
│   │   ├── ae_keras.py         # Arquitectura autoencoder
│   │   ├── ae_dmd.py           # Modelo AE+DMD
│   │   └── kovae.py            # KoVAE implementación
│   └── utils/
│       ├── download_era5.py    # Descarga Copernicus
│       ├── download_chirps.py  # Descarga CHIRPS
│       └── metrics.py          # MAE, RMSE, NSE
│
├── reports/
│   └── figures/                # 43 visualizaciones
│
├── doc/                        # Documentación técnica
│   ├── code_overview.md
│   └── PyDMD_paper.pdf
│
├── DATA_README.md              # Instrucciones de datos
├── GLOSARIO_TECNICO.md         # Glosario de términos
├── SINTESIS_FINAL.md           # Síntesis del proyecto
├── conda.yaml                  # Entorno reproducible
└── requirements.txt
```

---

## 8. Instalación y Reproducción

### 8.1 Requisitos

- Python 3.10.13
- TensorFlow 2.10.0 (GPU)
- CUDA 11.2 + cuDNN 8.1
- GPU NVIDIA (recomendado: RTX A4000 o superior)

### 8.2 Configuración del Entorno

```bash
# Crear entorno Conda
conda env create -f conda.yaml
conda activate capstone

# Verificar GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 8.3 Descarga de Datos

```bash
# ERA5 desde Copernicus CDS (requiere cuenta)
python src/utils/download_era5.py
python src/utils/merge_era5.py
python src/utils/merge_era5_advanced.py

# CHIRPS para validación
python src/utils/download_chirps.py
```

### 8.4 Ejecución de Notebooks

```bash
jupyter notebook notebooks/
```

**Orden recomendado:**
1. `01_EDA_Spatiotemporal.ipynb` - Análisis exploratorio
2. `02_Geoestadistica_Variogramas_Kriging.ipynb` - Variogramas y kriging
3. `03_AE_DMD_Training.ipynb` - Entrenamiento AE+DMD
4. `04_Advanced_Metrics.ipynb` - Métricas avanzadas
5. `05_KoVAE_Test.ipynb` - Modelo probabilístico
6. `06_Hyperparameter_Experiments.ipynb` - Optimización
7. `07_DMD_Interpretability.ipynb` - Interpretabilidad
8. `08_CHIRPS_Validation.ipynb` - Validacion satelital
9. `09_Benchmark_Comparison.ipynb` - Comparacion de modelos (ConvLSTM, AE+DMD, KoVAE)

---

## 9. Referencias

1. **Kutz, J.N. et al. (2016).** *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.* SIAM.

2. **Hersbach, H. et al. (2020).** *The ERA5 Global Reanalysis.* Quarterly Journal of the Royal Meteorological Society.

3. **Funk, C. et al. (2015).** *The Climate Hazards Infrared Precipitation with Stations (CHIRPS) Dataset.* Scientific Data.

4. **Cressie, N. & Wikle, C.K. (2011).** *Statistics for Spatio-Temporal Data.* Wiley.

5. **Lusch, B. et al. (2018).** *Deep learning for universal linear embeddings of nonlinear dynamics.* Nature Communications.

6. **Marchant & Silva (2024).** *AE+DMD para precipitaciones Chile.* Universidad del Desarrollo.

---

## 10. Contacto

- **Repositorio:** [github.com/Godoca2/capstone-project-pronostico-hibrido](https://github.com/Godoca2/capstone-project-pronostico-hibrido)
- **Issues:** GitHub Issues para reportar problemas
- **Email:** c.godoyd@udd.cl

---

**Ultima actualizacion:** 16 de Enero de 2026

*Universidad del Desarrollo - Magíster en Data Science*

