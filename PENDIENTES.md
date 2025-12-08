# Tareas Pendientes - Proyecto Capstone

**Última actualización**: 19 noviembre 2025 
**Progreso global**: 65% (3/5 fases completas) 
**Fase actual**: Fase 4 - Validación Satelital (15% completada)

---

## [PRIORIDAD ALTA] - Esta Semana (Semana 7)

### 1. Descargar Datos CHIRPS [En Espera]
**Tiempo estimado**: 2-4 horas (mayoría tiempo de descarga)

```bash
cd D:\11_Entorno_Desarrollo\UDD\captone_project\CAPSTONE_PROJECT
conda activate capstone
python src/utils/download_chirps.py
```

**Detalles**:
- Descarga ~2-4 GB de datos satelitales CHIRPS (2019-2020)
- Fuente: Climate Hazards Group InfraRed Precipitation
- Resolución: 0.05° (~5.5 km) vs ERA5 0.25° (~27.8 km)
- Output: `data/raw/chirps_chile_2019-01-01_2020-02-29.nc`
- Puede requerir múltiples intentos si la conexión se interrumpe
- Última vez se interrumpió al 30% (366 MB de 1.17 GB)

**Criterio de éxito**: Archivo NetCDF completo en `data/raw/`, sin errores de validación

---

### 2. Ejecutar Validación Satelital CHIRPS [En Espera]
**Tiempo estimado**: 1-2 horas

```bash
jupyter notebook notebooks/07_CHIRPS_Validation.ipynb
```

**Celdas a ejecutar** (8 celdas preparadas):
1. Imports y configuración
2. Cargar datos CHIRPS
3. Comparar resoluciones ERA5 vs CHIRPS
4. Mapas comparativos mensuales
5. Series temporales agregadas por región
6. Análisis de correlación espacial
7. Scatter plots ERA5 vs CHIRPS
8. Validación de predicciones contra CHIRPS

**Visualizaciones esperadas**:
- `chirps_vs_era5_comparison.png` - Mapas lado a lado
- `chirps_era5_correlation.png` - Correlación espacial
- `chirps_validation_scatter.png` - Dispersión
- `chirps_predictions_validation.png` - Validación independiente
- `chirps_bias_analysis.png` - Análisis de sesgo

**Criterio de éxito**: 8 celdas ejecutadas, 5+ figuras generadas, métricas de correlación calculadas

---

### 3. Comparación KoVAE vs AE+DMD [En Espera]
**Tiempo estimado**: 1 hora

**Archivo**: `notebooks/05_KoVAE_Test.ipynb` - Celda 11 (actualmente comentada)

**Pasos**:
1. Ejecutar `notebooks/02_DL_DMD_Forecast.ipynb` si no se ha guardado `forecast_results_2020.pkl`
2. Cargar resultados AE+DMD: `forecast_results['forecast_results']`
3. Descomentar y ejecutar celda 11
4. Comparar MAE horizonte h=1:
 - AE+DMD determinístico
 - KoVAE probabilístico
 - Diferencia porcentual
5. Generar visualización lado a lado

**Visualización esperada**:
- `kovae_vs_aedmd_comparison.png` - 3 paneles (Ground Truth / AE+DMD / KoVAE)

**Métricas a calcular**:
- MAE_AE+DMD vs MAE_KoVAE
- Cobertura intervalos de confianza KoVAE (¿95% real?)
- Valor agregado de incertidumbre para toma de decisiones

**Criterio de éxito**: Celda 11 ejecutada, 1 figura generada, comparación cuantitativa documentada

---

### 4. Commit Resultados Fase 4 [En Espera]
**Tiempo estimado**: 30 minutos

```bash
git add data/raw/chirps_*.nc data/models/kovae_trained/
git add reports/figures/chirps_*.png reports/figures/kovae_vs_aedmd_comparison.png
git add notebooks/05_KoVAE_Test.ipynb notebooks/08_CHIRPS_Validation.ipynb
git add README.md ROADMAP.md PENDIENTES.md
git commit -m "feat: Complete Phase 4 - CHIRPS validation and KoVAE vs AE+DMD comparison"
git push origin main
```

**Archivos a incluir**:
- Notebooks ejecutados (con outputs)
- Figuras CHIRPS (5+)
- Figura comparación KoVAE vs AE+DMD (1)
- Documentación actualizada (README, ROADMAP)

**Criterio de éxito**: Commit exitoso, GitHub actualizado con resultados completos Fase 4

---

## [PRIORIDAD MEDIA] - Próximas 2 Semanas (Semanas 8-9)

### 5. Paper Científico - Draft Completo [En Espera]
**Tiempo estimado**: 10-15 horas distribuidas

**Estructura propuesta** (IEEE/Springer):
1. **Abstract** (200 palabras) - [Pendiente] 30 min
2. **Introduction** (2 páginas) - [10% escrito] - [En Progreso] 3 horas
 - Contexto Chile: variabilidad climática
 - Problema: modelos numéricos limitados
 - Estado del arte: DL + geoestadística
 - Contribución: KoVAE + kriging weights (novel)
3. **Methodology** (3-4 páginas) - [Pendiente] 5 horas
 - Pipeline datos ERA5 (descarga, procesamiento)
 - Geoestadística: variogramas, kriging, pesos espaciales
 - AE+DMD baseline: arquitectura, loss function
 - KoVAE: operador Koopman, predicciones probabilísticas
 - Métricas: MAE, RMSE, NSE, Skill Scores
4. **Results** (3-4 páginas) - [Pendiente] 4 horas
 - Tabla comparativa baselines
 - Optimización hiperparámetros (13 configs)
 - Interpretabilidad DMD (23 modos)
 - KoVAE: reconstrucción (MAE=0.0029), predicciones multistep
 - Validación CHIRPS (correlación satelital)
5. **Discussion** (2 páginas) - [Pendiente] 2 horas
 - Ventajas KoVAE: incertidumbre, toma de decisiones
 - Limitaciones: 1 año datos (2020), resolución espacial
 - Aplicaciones: planificación hídrica, análisis riesgo
6. **Conclusions** (1 página) - [Pendiente] 1 hora
 - Resumen contribuciones
 - Trabajo futuro: multi-año, MODIS, cuencas hidrográficas
7. **References** (30-40 papers) - [Pendiente] 1 hora
 - Cressie & Wikle (2011), Lusch et al. (2018), Kutz et al. (2016)
 - Marchant & Silva (2024), Pérez & Zavala (2023) - UDD
 - ERA5, CHIRPS, PyDMD, TensorFlow

**Figuras clave a incluir**:
- Fig 1: Área de estudio (mapa Chile, macrozonas)
- Fig 2: Pipeline metodológico (diagrama flujo)
- Fig 3: Arquitectura KoVAE (encoder-koopman-decoder)
- Fig 4: Variograma y pesos espaciales
- Fig 5: Comparación baselines (tabla + gráfico barras)
- Fig 6: Modos DMD decodificados (top 3)
- Fig 7: KoVAE predicciones probabilísticas (IC 95%)
- Fig 8: Validación CHIRPS (mapas comparativos)

**Criterio de éxito**: Draft completo 15-20 páginas, listo para revisión tutor/comité

---

### 6. Presentación Defensa (Slides) [En Espera]
**Tiempo estimado**: 4-6 horas

**Estructura propuesta** (20-30 slides, 20-25 min presentación):
1. **Portada** (1 slide)
 - Título, autor, fecha, universidad
2. **Contexto y Problema** (2-3 slides)
 - Chile: variabilidad climática extrema
 - Impacto: agricultura, recursos hídricos
 - Limitación modelos actuales
3. **Objetivos** (1 slide)
 - Objetivo general: modelo híbrido espacio-temporal
 - Objetivos específicos: AE+DMD, KoVAE, validación
4. **Metodología** (5-6 slides)
 - Pipeline datos ERA5 (diagrama)
 - Geoestadística (variograma ejemplo)
 - AE+DMD (arquitectura esquemática)
 - KoVAE (operador Koopman, incertidumbre)
 - Métricas y baselines
5. **Resultados** (8-10 slides)
 - EDA: patrones estacionales por macrozona
 - AE+DMD: superación baselines (+10-17%)
 - Optimización: 13 configs, mejor MAE=1.934 mm/día
 - DMD: 23 modos estables, interpretación física
 - KoVAE: MAE=0.0029 reconstrucción, predicciones h=1-7
 - Validación CHIRPS: correlación satelital
 - Comparación KoVAE vs AE+DMD (incertidumbre)
6. **Discusión** (2-3 slides)
 - Ventajas: incertidumbre, toma decisiones
 - Limitaciones: 1 año, resolución
 - Aplicaciones reales
7. **Conclusiones** (1-2 slides)
 - Resumen contribuciones
 - Trabajo futuro
8. **Demo en Vivo** (5 min, 2-3 slides)
 - Ejecutar celda predicción KoVAE
 - Mostrar intervalos confianza
 - Visualización interactiva (opcional)
9. **Agradecimientos y Preguntas** (1 slide)

**Material de apoyo**:
- Video explicativo 5-10 min (opcional)
- Poster científico A0 (opcional)
- Notebook demo preparado

**Criterio de éxito**: Slides completos, ensayo 20-25 min cronometrado, demo funcional

---

## [PRIORIDAD BAJA] - Extensiones Opcionales (Post-Defensa)

### 7. Dataset 2019 Completo [En Espera]
**Tiempo estimado**: 3-4 horas

**Pasos**:
1. Modificar `src/utils/download_era5.py` para año 2019
2. Descargar 365 días (12 archivos mensuales ~1.5 GB)
3. Procesar con `merge_era5_advanced.py`
4. Combinar 2019+2020 (731 días total)
5. Split: 70% train (512 días) / 15% val (110 días) / 15% test (109 días)

**Beneficios**:
- Validación temporal robusta: train 2019 → test 2020
- Análisis estacional completo: 2 ciclos anuales
- Identificación eventos extremos históricos

**Criterio de éxito**: Dataset 731 días procesado, modelos re-entrenados, métricas comparadas

---

### 8. Validación Multi-Año (Cross-Year) [En Espera]
**Tiempo estimado**: 2-3 horas (requiere dataset 2019)

**Experimentos**:
1. Train 2019 (256 días) → Test 2020 (366 días)
2. Train 2020 (256 días) → Test 2019 (365 días)
3. Train 2019+2020 mixto → Test holdout

**Métricas**:
- Generalización temporal: MAE cross-year vs intra-year
- Análisis estacional: performance DJF, MAM, JJA, SON
- Skill scores por estación

**Criterio de éxito**: 3 experimentos completados, tabla comparativa generada

---

### 9. Análisis Eventos Extremos [En Espera]
**Tiempo estimado**: 3-4 horas

**Eventos a identificar** (2019-2020):
- Sistemas frontales invierno (Jun-Ago)
- Bloques altas presiones verano (Dic-Feb)
- Eventos extremos precipitación (>P95)
- Sequías prolongadas (<P05 por >30 días)

**Análisis**:
- Performance modelos en eventos extremos
- Intervalos confianza KoVAE para riesgo
- Detección temprana sistemas frontales

**Criterio de éxito**: 5-10 eventos identificados, análisis caso-por-caso documentado

---

### 10. Dashboard Interactivo (Streamlit) [En Espera]
**Tiempo estimado**: 6-8 horas

**Funcionalidades**:
- Upload modelo entrenado (.h5)
- Seleccionar fecha inicio
- Generar predicción h=1-7 días
- Visualizar mapas interactivos (Plotly)
- Mostrar intervalos confianza KoVAE
- Comparar múltiples modelos
- Exportar resultados CSV/PNG

**Stack**:
- Streamlit (frontend)
- TensorFlow (backend)
- Plotly (visualizaciones)
- Folium (mapas geoespaciales)

**Criterio de éxito**: Dashboard funcional localhost, demo grabado 5 min

---

### 11. API REST (FastAPI) [En Espera]
**Tiempo estimado**: 8-10 horas

**Endpoints**:
- `POST /predict`: Generar predicción
- `GET /models`: Listar modelos disponibles
- `GET /metrics`: Obtener métricas modelo
- `GET /history`: Histórico predicciones

**Infraestructura**:
- FastAPI + Uvicorn
- Dockerizado
- Documentación automática (Swagger)

**Criterio de éxito**: API desplegada localmente, pruebas Postman exitosas

---

## [INFO] Resumen de Prioridades

| Tarea | Prioridad | Tiempo | Fase | Necesario para Defensa |
|-------|-----------|--------|------|------------------------|
| Descargar CHIRPS | [PRIORIDAD ALTA] ALTA | 2-4h | 4 | [OK] SÍ |
| Validación CHIRPS | [PRIORIDAD ALTA] ALTA | 1-2h | 4 | [OK] SÍ |
| Comparación KoVAE vs AE+DMD | [PRIORIDAD ALTA] ALTA | 1h | 4 | [OK] SÍ |
| Commit Resultados | [PRIORIDAD ALTA] ALTA | 30m | 4 | [OK] SÍ |
| Paper Draft | [PRIORIDAD MEDIA] MEDIA | 10-15h | 5 | [OK] SÍ |
| Presentación Slides | [PRIORIDAD MEDIA] MEDIA | 4-6h | 5 | [OK] SÍ |
| Dataset 2019 | [PRIORIDAD BAJA] BAJA | 3-4h | 4 | [ERROR] NO (opcional) |
| Validación Multi-Año | [PRIORIDAD BAJA] BAJA | 2-3h | 4 | [ERROR] NO (opcional) |
| Eventos Extremos | [PRIORIDAD BAJA] BAJA | 3-4h | 4 | [ERROR] NO (opcional) |
| Dashboard Streamlit | [PRIORIDAD BAJA] BAJA | 6-8h | 5 | [ERROR] NO (opcional) |
| API REST | [PRIORIDAD BAJA] BAJA | 8-10h | 5 | [ERROR] NO (opcional) |

**Total horas críticas (prioridad alta+media)**: 19-25 horas 
**Total horas opcionales (prioridad baja)**: 22-31 horas

---

## [OK] Criterios de Completitud para Defensa

### Mínimo Viable (Aprobación)
- [OK] 3 fases completadas (1, 2, 3)
- [OK] 7 notebooks ejecutados
- [OK] 30+ figuras científicas
- [OK] Documentación completa
- [En Progreso] Validación independiente (CHIRPS) - EN PROGRESO

### Objetivo Distinción (7.0)
- [OK] Todo lo anterior
- [En Progreso] Validación satelital ejecutada
- [En Espera] Paper draft completo
- [En Espera] Presentación slides completa
- [OK] KoVAE implementado y validado

### Excelencia (7.0)
- [OK] Todo lo anterior
- [OK] Contribución metodológica novel (KoVAE + kriging)
- [En Espera] Resultados publicables
- [ERROR] Multi-año (opcional)
- [ERROR] Dashboard/API (opcional)

---

**Estado actual**: **Camino a Distinción (85-90%)** 
**Próximo hito**: Completar Fase 4 (CHIRPS + comparación) - Semana 7 
**Deadline sugerido defensa**: Semana 12 (finales noviembre 2025)
