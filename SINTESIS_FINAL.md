# SÍNTESIS FINAL DEL PROYECTO
## Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile

**Autor:** César Godoy Delaigue  
**Institución:** Universidad del Desarrollo (UDD)  
**Fecha de Cierre:** 2 de Enero de 2026  
**Estado:** ✅ COMPLETADO

---

## 1. RESUMEN EJECUTIVO

Este proyecto desarrolló un sistema híbrido de pronóstico de precipitaciones para Chile continental combinando técnicas de **Deep Learning** (Autoencoders Convolucionales), **Análisis de Sistemas Dinámicos** (DMD - Dynamic Mode Decomposition) y **Geoestadística** (Kriging y Variogramas).

### Logros Principales

| Aspecto | Resultado |
|---------|-----------|
| **MAE del modelo** | 1.763 mm/día (±0.04) |
| **Mejora vs Persistence** | +7.1% |
| **Mejora vs Climatología** | +12.9% |
| **Compresión espacial** | 100.3× (6,437 → 64 dim) |
| **Cobertura temporal** | 366 días (2020 completo) |
| **Validación externa** | CHIRPS satelital |

---

## 2. ARQUITECTURA DEL SISTEMA

### 2.1 Pipeline Completo

```
ERA5 Reanalysis (0.25°)
        ↓
   Preprocesamiento
   (Agregación diaria, normalización)
        ↓
┌───────────────────────────────────┐
│     AUTOENCODER CONVOLUCIONAL     │
│  - Encoder: Conv2D dilatadas      │
│  - Latent: 64 dimensiones         │
│  - Decoder: Conv2DTranspose       │
│  - Loss: Weighted MSE (Kriging)   │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│            DMD                    │
│  - 23 modos estables              │
│  - Proyección temporal A^h        │
│  - Horizontes: 1, 3, 7 días       │
└───────────────────────────────────┘
        ↓
   Pronóstico Final
        ↓
   Validación CHIRPS
```

### 2.2 Modelos Implementados

1. **AE+DMD (Baseline):** Autoencoder con DMD para proyección temporal
2. **KoVAE (Experimental):** Variational Autoencoder con operador Koopman

---

## 3. RESULTADOS FINALES VALIDADOS

### 3.1 Métricas Globales (Notebook 04)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| MAE | 1.763 mm/día | Error promedio absoluto |
| RMSE | 4.305 mm/día | Penaliza errores grandes |
| Skill vs Persistence | +7.1% | Supera al baseline naive |
| Skill vs Climatología | +12.9% | Supera al promedio histórico |

### 3.2 Desempeño por Macrozona

| Macrozona | Latitudes | MAE (mm/día) | Característica |
|-----------|-----------|--------------|----------------|
| **Norte** | -17° a -30° | 0.89 | Menor error (árido) |
| **Centro** | -30° a -40° | 1.92 | Mediterráneo |
| **Sur** | -40° a -56° | 3.41 | Mayor variabilidad |

### 3.3 Validación CHIRPS (Notebook 08)

| Comparación | MAE (mm/día) | Diferencia |
|-------------|--------------|------------|
| Modelo vs CHIRPS | 1.8708 | Referencia |
| ERA5 vs CHIRPS | 2.0112 | +7% |
| **Modelo mejora ERA5** | - | ✅ Confirmado |

**Rendimiento Regional vs CHIRPS:**
- Norte: +24.3% mejor que ERA5
- Sur: +28.3% mejor que ERA5  
- Centro: -24.0% (margen de mejora)

### 3.4 Análisis DMD (Notebook 07)

- **Modos estables identificados:** 23
- **Cobertura temporal:** 365 días
- **Frecuencias dominantes:** Estacional (anual) + sinóptica (7-10 días)

### 3.5 KoVAE Experimental (Notebook 05)

| Métrica | Valor |
|---------|-------|
| Reconstruction MAE | 0.0029 mm/día |
| Train Loss | 0.009911 |
| Validation Loss | 0.027073 |
| Capacidad probabilística | ✅ Incertidumbre por región |

---

## 4. CONTRIBUCIONES GEOESTADÍSTICAS

### 4.1 Variograma Ajustado (Modelo Esférico)

| Parámetro | Valor | Interpretación |
|-----------|-------|----------------|
| **Range** | 8.15° (~913 km) | Alcance de correlación espacial |
| **Sill** | 23.45 mm²/día² | Varianza total |
| **Nugget** | 0.0 | Sin ruido sub-grid |

### 4.2 Aplicaciones del Variograma

1. **Diseño de CNN:** Receptive field ≥ 33 celdas (cumplido: ~40)
2. **Kriging:** Interpolación óptima con R² = 0.9923
3. **Loss ponderada:** Pesos espaciales basados en varianza kriging

---

## 5. ESTRUCTURA FINAL DEL REPOSITORIO

```
CAPSTONE_PROJECT/
├── data/
│   ├── models/
│   │   ├── autoencoder_geostat.h5    # AE principal
│   │   ├── encoder_geostat.h5        # Encoder separado
│   │   ├── kovae_trained/            # KoVAE completo
│   │   ├── training_metrics.csv      # Métricas de entrenamiento
│   │   ├── ablation/                 # Experimentos γ=0, γ=0.1
│   │   └── ablation_long/            # Experimentos extendidos
│   └── processed/
│       ├── era5_precipitation_chile_full.nc
│       └── era5_precipitation_chile_daily.csv
│
├── notebooks/
│   ├── 01_EDA_Spatiotemporal.ipynb
│   ├── 02_Geoestadistica_Variogramas_Kriging.ipynb
│   ├── 03_AE_DMD_Training.ipynb
│   ├── 04_Advanced_Metrics.ipynb
│   ├── 05_KoVAE_Test.ipynb
│   ├── 06_Hyperparameter_Experiments.ipynb
│   ├── 07_DMD_Interpretability.ipynb
│   └── 08_CHIRPS_Validation.ipynb
│
├── reports/
│   ├── figures/                      # 65+ visualizaciones
│   ├── ablation_report.md
│   ├── chirps_validation_summary.md
│   ├── eda_summary.md
│   └── metrics_eval.csv
│
├── src/
│   ├── models/
│   │   ├── ae_keras.py
│   │   ├── ae_dmd.py
│   │   └── kovae.py
│   ├── geo/
│   │   └── variogram_kriging.py
│   └── utils/
│
└── doc/
    └── code_overview.md
```

---

## 6. NOTEBOOKS - RESUMEN DE EJECUCIÓN

| # | Notebook | Celdas | Estado | Output Principal |
|---|----------|--------|--------|------------------|
| 01 | EDA_Spatiotemporal | 23 | ✅ | Análisis exploratorio |
| 02 | Geoestadistica | 29 | ✅ | Variograma + Kriging |
| 03 | AE_DMD_Training | 54 | ✅ | Modelo entrenado |
| 04 | Advanced_Metrics | 18 | ✅ | MAE=1.763 mm/día |
| 05 | KoVAE_Test | 26 | ✅ | KoVAE probabilístico |
| 06 | Hyperparameter | 21 | ✅ | 13 configuraciones |
| 07 | DMD_Interpretability | 37 | ✅ | 23 modos estables |
| 08 | CHIRPS_Validation | 15 | ✅ | Validación externa |

---

## 7. LECCIONES APRENDIDAS

### 7.1 Éxitos

1. **Integración Geoestadística-DL:** El variograma guía efectivamente el diseño de CNN
2. **Compresión eficiente:** 100× reducción con reconstrucción de alta fidelidad
3. **Validación robusta:** CHIRPS confirma generalización del modelo
4. **Reproducibilidad:** Pipeline completo con seeds fijos y MLflow

### 7.2 Desafíos Superados

1. **Determinismo GPU:** Resuelto con `TF_DETERMINISTIC_OPS=1` y Conv2DTranspose
2. **Escala de precipitación:** Normalización cuidadosa y transformación log
3. **Zona Centro:** Identificada como área de mejora (variabilidad mediterránea)

### 7.3 Trabajo Futuro

1. **Ensemble AE+DMD + KoVAE:** Combinar predicciones determinísticas y probabilísticas
2. **Asimilación de datos:** Integrar observaciones en tiempo real
3. **Resolución mejorada:** Downscaling a 0.05° usando CHIRPS como target
4. **Extensión temporal:** Entrenar con múltiples años (2015-2023)

---

## 8. MÉTRICAS DE CALIDAD DEL PROYECTO

| Aspecto | Cumplimiento |
|---------|--------------|
| Reproducibilidad | ✅ 100% (seeds, MLflow) |
| Documentación | ✅ 100% (8 archivos .md) |
| Validación externa | ✅ CHIRPS satelital |
| Código modular | ✅ src/ estructurado |
| Visualizaciones | ✅ 65+ figuras |
| Tests | ⚠️ Parcial (scripts de diagnóstico) |

---

## 9. CONCLUSIÓN

El proyecto **"Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile"** ha sido completado exitosamente, demostrando que:

1. **La combinación AE+DMD+Geoestadística es viable** para pronóstico de precipitación a escala regional

2. **El modelo supera baselines clásicos** (Persistence +7.1%, Climatología +12.9%)

3. **La validación con CHIRPS confirma** que el modelo mejora sobre ERA5 crudo en la mayoría de las regiones

4. **KoVAE ofrece capacidades probabilísticas** útiles para cuantificar incertidumbre

El sistema desarrollado representa una contribución metodológica al campo de la predicción meteorológica híbrida, combinando principios físicos (geoestadística) con aprendizaje profundo de manera interpretable.

---

## 10. REFERENCIAS

1. Kutz, J.N. et al. (2016). *Dynamic Mode Decomposition*. SIAM.
2. Hersbach, H. et al. (2020). *The ERA5 Global Reanalysis*. QJRMS.
3. Funk, C. et al. (2015). *The CHIRPS Dataset*. Scientific Data.
4. Cressie, N. (1993). *Statistics for Spatial Data*. Wiley.

---

**Proyecto completado el 2 de Enero de 2026**

*César Godoy Delaigue - Universidad del Desarrollo*
