# Guía de Datos del Proyecto

## Archivos de Datos No Incluidos en el Repositorio

Por limitaciones de tamaño de GitHub (límite 100MB por archivo), los siguientes archivos de datos **NO están incluidos** en este repositorio pero son necesarios para ejecutar los notebooks:

### Datos Requeridos

#### 1. **ERA5 Precipitation Data**
- **Archivo:** `data/processed/era5_precipitation_chile_full.nc`
- **Tamaño:** ~45 MB
- **Descripción:** Precipitación diaria ERA5 para Chile Continental (2020)
- **Cobertura:** 17°S-56°S, 67°W-75°W, resolución 0.25°
- **Cómo obtener:**
  ```bash
  # Ejecutar notebook de descarga
  python src/utils/download_era5.py --year 2020 --variable total_precipitation
  ```
  O descargar manualmente desde:
  - **ERA5 Land Hourly:** https://cds.climate.copernicus.eu/
  - Registrarse en CDS API
  - Usar script `src/utils/download_era5.py`

#### 2. **CHIRPS Satellite Data** (Validación)
- **Archivo:** `data/external/chirps/chirps-v2.0.2019.days_p05.nc`
- **Tamaño:** ~366 MB
- **Descripción:** Datos satelitales CHIRPS para validación cruzada
- **Resolución:** 0.05° (~5.5 km)
- **Cómo obtener:**
  ```bash
  # Descarga desde servidor CHIRPS
  wget ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.2019.days_p05.nc -P data/external/chirps/
  ```
  O desde: https://www.chc.ucsb.edu/data/chirps

#### 3. **Kriging Interpolation Output**
- **Archivo:** `data/processed/kriging_precipitation_june_2020.nc`
- **Tamaño:** ~60 MB
- **Descripción:** Resultado de interpolación kriging (generado por Notebook 02)
- **Cómo generar:**
  ```bash
  # Ejecutar Notebook 02
  jupyter notebook notebooks/02_Geoestadistica_Variogramas_Kriging.ipynb
  ```
  Este archivo se genera automáticamente al ejecutar todas las celdas.

#### 4. **Modelos Entrenados** (Disponibles en repositorio)
- **Archivos:** 
  - `data/models/autoencoder_geostat.h5` (~4 MB) - AE+DMD baseline
  - `data/models/encoder_geostat.h5` (~2 MB) - Encoder standalone
  - `data/models/kovae_trained/kovae_full.h5` (~42 MB) - KoVAE completo
  - `data/models/kovae_trained/encoder.h5` - Encoder probabilístico
  - `data/models/kovae_trained/decoder.h5` - Decoder generativo
  - `data/models/kovae_trained/koopman_matrix.npy` - Matriz K 64×64
  - `data/models/kovae_trained/config.pkl` - Configuración
  - `data/models/training_metrics.csv` - Métricas de entrenamiento
- **Descripción:** Pesos entrenados de modelos (pipeline completo ejecutado)
- **Métricas validadas:**
  - AE+DMD: MAE=1.763 mm/día, mejora +7.1% vs Persistence
  - KoVAE: MAE reconstrucción=0.0029 mm/día
- **Cómo regenerar:**
  ```bash
  # Entrenar modelos desde cero
  jupyter notebook notebooks/03_AE_DMD_Training.ipynb  # AE+DMD
  jupyter notebook notebooks/05_KoVAE_Test.ipynb       # KoVAE
  ```

---

## Setup Rápido (Reproducción Completa)

### Opción 1: Desde Cero (Recomendado para Reproducibilidad)

```bash
# 1. Clonar repositorio
git clone https://github.com/Godoca2/Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile.git
cd Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile

# 2. Crear ambiente conda
conda env create -f conda.yaml
conda activate capstone

# 3. Instalar dependencias adicionales
pip install -r requirements.txt

# 4. Descargar datos ERA5
python src/utils/download_era5.py --year 2020 --variable total_precipitation --region chile

# 5. (Opcional) Descargar CHIRPS para validación
wget ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.2019.days_p05.nc -P data/external/chirps/

# 6. Ejecutar pipeline completo
jupyter notebook notebooks/
# Ejecutar en orden: 01 → 02 → 03 → 04 → ... → 08
```

### Opción 2: Con Datos Pre-procesados (Más Rápido)

Si tienes acceso a los datos ya descargados (por ejemplo, compartidos por Google Drive):

```bash
# 1. Clonar repositorio
git clone https://github.com/Godoca2/Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile.git
cd Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile

# 2. Descargar datos desde Google Drive/OneDrive/etc
# (Link provisto por el autor del proyecto)
# Descomprimir en las carpetas correspondientes:
unzip data_capstone.zip -d data/

# 3. Crear ambiente y ejecutar
conda env create -f conda.yaml
conda activate capstone
jupyter notebook notebooks/
```

---

## Descarga de Datos Pre-procesados

**Para revisores del proyecto (Profesor Guía, Comisión Evaluadora):**

Los datos completos están disponibles en:
- **Google Drive:** [LINK_A_COMPARTIR]
- **OneDrive UDD:** [LINK_A_COMPARTIR]
- **Tamaño total:** ~500 MB (comprimido)

Contenido del archivo `data_capstone.zip`:
```
data/
├── external/
│   └── chirps/
│       └── chirps-v2.0.2019.days_p05.nc (366 MB)
├── processed/
│   ├── era5_precipitation_chile_full.nc (45 MB)
│   └── kriging_precipitation_june_2020.nc (60 MB)
└── models/
    ├── autoencoder_geostat.h5 (4 MB)
    ├── encoder_geostat.h5 (2 MB)
    └── kovae_trained/
        └── kovae_full.h5 (65 MB)
```

---

## Verificación de Datos

Para verificar que todos los datos necesarios están presentes:

```python
import os
from pathlib import Path

# Lista de archivos requeridos
required_files = [
    "data/processed/era5_precipitation_chile_full.nc",
    "data/external/chirps/chirps-v2.0.2019.days_p05.nc",
    "data/processed/kriging_precipitation_june_2020.nc",  # Generado por Notebook 02
]

# Verificar existencia
for file_path in required_files:
    if Path(file_path).exists():
        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        print(f"✅ {file_path} ({size_mb:.2f} MB)")
    else:
        print(f"❌ {file_path} - NO ENCONTRADO")
```

---

## Estructura de Datos Esperada

```
CAPSTONE_PROJECT/
├── data/
│   ├── external/
│   │   └── chirps/
│   │       └── *.nc (datos satelitales)
│   ├── processed/
│   │   ├── era5_precipitation_chile_full.nc
│   │   ├── kriging_precipitation_june_2020.nc
│   │   ├── metrics_summary.csv
│   │   └── kovae_worst_cells_examples.csv
│   ├── models/
│   │   ├── autoencoder_geostat.h5 (AE+DMD baseline)
│   │   ├── encoder_geostat.h5
│   │   ├── training_metrics.csv
│   │   ├── kovae_trained/
│   │   │   ├── kovae_full.h5
│   │   │   ├── encoder.h5
│   │   │   ├── decoder.h5
│   │   │   ├── koopman_matrix.npy
│   │   │   └── config.pkl
│   │   ├── ablation/
│   │   └── ablation_long/
│   └── raw/
│       └── precipitation_data.npy
├── notebooks/
│   ├── 01_EDA_Spatiotemporal.ipynb
│   ├── 02_Geoestadistica_Variogramas_Kriging.ipynb
│   ├── 03_AE_DMD_Training.ipynb
│   ├── 04_Advanced_Metrics.ipynb
│   ├── 05_KoVAE_Test.ipynb
│   ├── 06_Hyperparameter_Experiments.ipynb
│   ├── 07_DMD_Interpretability.ipynb
│   └── 08_CHIRPS_Validation.ipynb
├── reports/
│   ├── figures/ (65+ figuras generadas)
│   ├── chirps_validation_summary.md
│   ├── ablation_report.md
│   └── metrics_eval.csv
└── src/
    └── utils/
        ├── download_era5.py
        ├── download_chirps.py
        └── metrics.py
```

---

## Requisitos del Sistema

### Mínimos
- **Python:** 3.9+
- **RAM:** 16 GB
- **Almacenamiento:** 2 GB libres
- **GPU:** Opcional (CPU funciona pero más lento)

### Recomendados (para entrenamiento)
- **Python:** 3.10
- **RAM:** 32 GB
- **GPU:** NVIDIA con 8+ GB VRAM (CUDA 11.2+)
- **Almacenamiento:** 5 GB libres

---

## Troubleshooting

### Error: "FileNotFoundError: era5_precipitation_chile_full.nc"
**Solución:** Descargar datos ERA5 siguiendo instrucciones arriba.

### Error: "MemoryError durante carga de datos"
**Solución:** 
- Cerrar otras aplicaciones
- Usar chunks en xarray: `ds = xr.open_dataset(file, chunks={'time': 50})`

### Error: "ModuleNotFoundError: No module named 'pydmd'"
**Solución:** 
```bash
conda activate capstone
pip install pydmd
```

---

## Contacto

**Autor:** César Godoy Delaigue  
**Institución:** Universidad del Desarrollo (UDD)  
**Proyecto:** Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile  
**Email:** [cesar.godoy@udd.cl]

Para acceso a datos pre-procesados o consultas sobre reproducción, contactar al autor.

---

## 📄 Licencia

Los datos ERA5 y CHIRPS están sujetos a sus respectivas licencias:
- **ERA5:** Copernicus Climate Change Service (C3S)
- **CHIRPS:** UC Santa Barbara Climate Hazards Group

El código de este proyecto está bajo licencia MIT (ver LICENSE).
