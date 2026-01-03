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

### Resultados Principales

| Métrica | Valor |
|---------|-------|
| **MAE** | 1.763 mm/día (±0.04) |
| **Mejora vs Persistence** | +7.1% |
| **Mejora vs Climatología** | +12.9% |
| **Compresión espacial** | 100.3× (6,437 → 64 dim) |
| **Validación CHIRPS** | +7% mejor que ERA5 crudo |

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

### 4.1 Pipeline del Sistema

```
ERA5 Reanalysis (0.25°, horario)
            ↓
    Preprocesamiento
    (Agregación diaria, normalización)
            ↓
┌─────────────────────────────────────┐
│   AUTOENCODER CONVOLUCIONAL         │
│   - Encoder: Conv2D dilatadas       │
│   - Latent: 64 dimensiones          │
│   - Decoder: Conv2DTranspose        │
│   - Loss: Weighted MSE (Kriging)    │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│              DMD                    │
│   - 23 modos estables               │
│   - Proyección temporal A^h         │
│   - Horizontes: 1, 3, 7 días        │
└─────────────────────────────────────┘
            ↓
      Pronóstico Final
            ↓
    Validación CHIRPS (0.05°)
```

### 4.2 Arquitectura del Autoencoder

**Diseño informado por variogramas:**

- **Encoder:** Dilated CNN (dilations=[1,2,4,8])
  - Receptive field ~40 celdas (cumple range 8.15° del variograma)
  - MaxPooling 2×2 (3 capas) → compresión espacial
  - Bottleneck: 64-dim latent space

- **Decoder:** Conv2DTranspose simétrico
  - UpSampling 2×2 (3 capas)
  - Output: (157, 41, 1)

- **Loss function:** Weighted MSE
  - Pesos = 1 / (varianza_kriging + ε)
  - Penaliza más errores en zonas de alta confianza

### 4.3 Análisis Geoestadístico

| Parámetro | Valor | Interpretación |
|-----------|-------|----------------|
| **Range** | 8.15° (~913 km) | Alcance de correlación espacial |
| **Sill** | 23.45 mm²/día² | Varianza total |
| **Nugget** | 0.0 | Sin ruido sub-grid |

**Aplicaciones:**
1. Diseño del receptive field de CNN (≥33 celdas)
2. Kriging para interpolación óptima
3. Pesos espaciales en función de pérdida

### 4.4 KoVAE (Modelo Probabilístico)

Implementación del Kolmogorov-Arnold Variational Autoencoder con operador Koopman:

- **Arquitectura:** Encoder probabilístico (μ, log σ²) → Koopman Layer (64×64) → Decoder
- **Pérdida compuesta:** L_recon + β·KL + γ·L_koopman
- **Ventajas:** Cuantificación de incertidumbre, intervalos de confianza 95%

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

| Conjunto | Proporción | Secuencias |
|----------|------------|------------|
| Train | 70% | 251 |
| Validation | 15% | 53 |
| Test | 15% | 55 |

---

## 6. Resultados

### 6.1 Métricas Globales

| Modelo | MAE (mm/día) | RMSE (mm/día) | vs Persistence | vs Climatología |
|--------|--------------|---------------|----------------|-----------------|
| **AE+DMD** | 1.763 | 4.305 | +7.1% | +12.9% |
| **KoVAE** | 0.0029* | 0.0055* | - | - |

*Métricas de reconstrucción

### 6.2 Desempeño Regional

| Macrozona | Latitudes | MAE (mm/día) | vs CHIRPS |
|-----------|-----------|--------------|-----------|
| **Norte** | -17° a -30° | 0.89 | +24.3% mejor |
| **Centro** | -30° a -40° | 1.92 | -24.0% |
| **Sur** | -40° a -56° | 3.41 | +28.3% mejor |

### 6.3 Análisis DMD

- **Modos estables identificados:** 23 (100% estables)
- **Cobertura temporal:** 365 días
- **Interpretación física:**
  - Modo #1: Estado base climatológico nacional
  - Modo #2: Separación fenómenos regionales (Alta de Bolivia)
  - Modos #3-5: Variabilidad regional diferenciada

### 6.4 Validación CHIRPS

| Comparación | MAE (mm/día) |
|-------------|--------------|
| Modelo vs CHIRPS | 1.8708 |
| ERA5 vs CHIRPS | 2.0112 |
| **Mejora modelo** | **+7%** |

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
│   └── 08_CHIRPS_Validation.ipynb
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
8. `08_CHIRPS_Validation.ipynb` - Validación satelital

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

**Última actualización:** 3 de Enero de 2026

*Universidad del Desarrollo - Magíster en Data Science*

