"""
merge_era5.py
--------------
Combina múltiples archivos NetCDF (ERA5) en un solo Dataset y lo exporta
como NetCDF y CSV listos para análisis.

Autor: César Godoy Delaigue
Magíster Data Science UDD - 2025
"""

import xarray as xr
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def merge_era5_files(variable: str = "total_precipitation"):
 """
 Combina todos los archivos .nc en data/raw en un solo dataset.
 """
 files = sorted(list(RAW_DIR.glob("era5_precipitation_chile_*.nc")))
 if not files:
 raise FileNotFoundError("[ERROR] No se encontraron archivos NetCDF en data/raw/.")
 
 print(f"[INFO] Encontrados {len(files)} archivos NetCDF. Iniciando combinación...")
 
 # Combina todos los datasets (usa concatenación por tiempo)
 ds = xr.open_mfdataset(
 files,
 combine="by_coords",
 parallel=True,
 chunks={"time": 500}
 )

 # Selecciona la variable principal
 if variable not in ds.data_vars:
 variable = list(ds.data_vars)[0]
 print(f"[AVISO] Variable '{variable}' seleccionada automáticamente.")
 ds = ds[[variable]]
 
 # Ordenar dimensiones
 ds = ds.sortby(["time", "latitude", "longitude"])

 out_nc = OUT_DIR / "era5_precipitation_chile_full.nc"
 ds.to_netcdf(out_nc)
 print(f"[OK] Archivo combinado guardado: {out_nc}")

 # Convertir a DataFrame resumido
 print(" Generando DataFrame CSV (promedios diarios)...")
 df = ds[variable].to_dataframe().reset_index()
 df["date"] = pd.to_datetime(df["time"]).dt.date
 df_daily = df.groupby(["date", "latitude", "longitude"])[variable].sum().reset_index()
 
 out_csv = OUT_DIR / "era5_precipitation_chile_daily.csv"
 df_daily.to_csv(out_csv, index=False)
 print(f"[OK] DataFrame guardado: {out_csv}")
 
 return ds, df_daily

if __name__ == "__main__":
 ds, df_daily = merge_era5_files("total_precipitation")
 print(ds)
 print(df_daily.head())
 
 
 
 
"""

 Cómo usarlo

Después de haber corrido el script download_era5.py y tener varios .nc en data/raw/, simplemente ejecuta:

python src/utils/merge_era5.py

[INFO] Esto producirá dos archivos:

data/processed/era5_precipitation_chile_full.nc
data/processed/era5_precipitation_chile_daily.csv

Y te dejará el DataFrame df_daily listo para análisis o entrada a tu notebook 01_eda.ipynb.

 Ventajas del flujo completo

[OK] Descarga reproducible (por año, automatizada).
[OK] Combinación eficiente con xarray.open_mfdataset (usa chunks para bajo consumo de RAM).
[OK] Conversión a formato NetCDF consolidado + CSV limpio.
[OK] Preparado para entrar directo al módulo data_prep.py o para entrenar tu AE/DMD.

""" 
