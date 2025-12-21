"""
download_era5.py
----------------
Descarga automática de datos ERA5 (precipitación total) desde el
Copernicus Climate Data Store (CDS) utilizando la API oficial.

Autor: César Godoy Delaigue
Magíster Data Science UDD - 2025
"""

import cdsapi
from pathlib import Path

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================

# Carpeta de destino
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variables de configuración
DATASET = "reanalysis-era5-single-levels"
VARIABLES = ["total_precipitation"]
YEARS = list(range(1980, 2023)) # 1980–2022
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(0, 24)]

# Recorte espacial (Chile)
# Formato: [Norte, Oeste, Sur, Este]
AREA_CHILE = [-17, -75, -56, -66]

# ==========================
# DESCARGA DE DATOS
# ==========================

def download_era5_precipitation(year: int):
 """
 Descarga los datos ERA5 de un año específico para Chile.

 Parámetros
 ----------
 year : int
 Año a descargar.
 """
 c = cdsapi.Client()
 output_file = OUTPUT_DIR / f"era5_precipitation_chile_{year}.nc"

 if output_file.exists():
 print(f"[OK] {output_file.name} ya existe. Saltando descarga.")
 return

 print(f"[INFO] Descargando ERA5 precipitación para el año {year}...")

 try:
 c.retrieve(
 DATASET,
 {
 "product_type": "reanalysis",
 "variable": VARIABLES,
 "year": str(year),
 "month": MONTHS,
 "day": DAYS,
 "time": HOURS,
 "area": AREA_CHILE,
 "format": "netcdf",
 },
 str(output_file),
 )
 print(f"[OK] Descarga completada: {output_file}")
 except Exception as e:
 print(f"[ERROR] Error al descargar el año {year}: {e}")

def batch_download(start_year: int = 1980, end_year: int = 2022):
 """
 Descarga en lote varios años consecutivos de ERA5.
 """
 for year in range(start_year, end_year + 1):
 download_era5_precipitation(year)

if __name__ == "__main__":
 # Ejecución directa (descarga por años)
 batch_download(1980, 2022)
 
 
 
"""
 Configuración previa: credenciales Copernicus

Antes de ejecutar el script por primera vez, debes tener un archivo oculto en tu usuario:

 ~/.cdsapirc (en Windows, C:\Users\<tu_usuario>\.cdsapirc)

url: https://cds.climate.copernicus.eu/api/v2
key: <tu_usuario>:<tu_token>

 Puedes obtener el key/token en tu cuenta de Copernicus CDS

 Ejecución desde VS Code

Abre la terminal en la raíz del proyecto y corre:

python src/utils/download_era5.py

Esto descargará automáticamente los archivos .nc año por año en:

data/raw/era5_precipitation_chile_1980.nc
data/raw/era5_precipitation_chile_1981.nc
...
data/raw/era5_precipitation_chile_2022.nc

 Si interrumpes el proceso, simplemente vuelve a ejecutar el script; detectará los archivos ya descargados y continuará desde donde quedó.

 Opcional: usar rango más corto para pruebas

Durante tus pruebas, puedes cambiar la línea final:

batch_download(2020, 2022)

Así descargas solo tres años (mucho más rápido, ideal para testear el flujo y el notebook de EDA).

""" 
