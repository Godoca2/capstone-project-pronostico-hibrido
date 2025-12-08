"""
merge_era5_advanced.py
----------------------
Combina archivos NetCDF (ERA5) y genera estadísticas agregadas por macrozonas de Chile.
Exporta resultados en NetCDF, CSV y figuras visuales.

Autor: César Godoy Delaigue
Magíster Data Science UDD - 2025
Colaboración: FlowHydro Consultores en Recursos Hídricos
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")
for d in [OUT_DIR, FIG_DIR]:
 d.mkdir(parents=True, exist_ok=True)

# === DEFINICIÓN DE MACROZONAS CLIMÁTICAS DE CHILE ===
ZONES = {
 "norte": {"lat_range": (-24, -17), "lon_range": (-70, -66)},
 "centro": {"lat_range": (-36, -24), "lon_range": (-74, -69)},
 "sur": {"lat_range": (-56, -36), "lon_range": (-75, -71)},
}

def merge_era5_files(variable: str = "total_precipitation") -> xr.Dataset:
 """
 Combina todos los archivos .nc en data/raw en un solo Dataset.
 """
 files = sorted(list(RAW_DIR.glob("era5_precipitation_chile_*.nc")))
 if not files:
 raise FileNotFoundError("[ERROR] No se encontraron archivos NetCDF en data/raw/.")
 print(f"[INFO] {len(files)} archivos encontrados. Combinando con xarray...")

 ds = xr.open_mfdataset(files, combine="by_coords", parallel=True, chunks={"time": 500})

 if variable not in ds.data_vars:
 variable = list(ds.data_vars)[0]
 print(f"[AVISO] Variable '{variable}' seleccionada automáticamente.")
 ds = ds[[variable]]
 ds = ds.sortby(["time", "latitude", "longitude"])

 out_nc = OUT_DIR / "era5_precipitation_chile_full.nc"
 ds.to_netcdf(out_nc)
 print(f"[OK] Dataset combinado guardado: {out_nc}")
 return ds

def compute_daily_means(ds: xr.Dataset, variable: str = "total_precipitation") -> pd.DataFrame:
 """
 Convierte un dataset horario a agregaciones diarias.
 """
 print(" Calculando promedios diarios...")
 df = ds[variable].to_dataframe().reset_index()
 df["date"] = pd.to_datetime(df["time"]).dt.date
 df_daily = df.groupby(["date", "latitude", "longitude"])[variable].sum().reset_index()
 out_csv = OUT_DIR / "era5_precipitation_chile_daily.csv"
 df_daily.to_csv(out_csv, index=False)
 print(f"[OK] Datos diarios exportados: {out_csv}")
 return df_daily

def filter_by_zone(df: pd.DataFrame, zone: str) -> pd.DataFrame:
 """
 Filtra un DataFrame por macrozona (norte, centro o sur).
 """
 z = ZONES[zone]
 df_z = df[
 (df["latitude"].between(*z["lat_range"])) &
 (df["longitude"].between(*z["lon_range"]))
 ].copy()
 print(f" {zone.upper()}: {len(df_z):,} registros seleccionados.")
 return df_z

def compute_temporal_aggregates(df: pd.DataFrame, variable: str = "total_precipitation"):
 """
 Calcula promedios anuales y mensuales por macrozona.
 """
 df["date"] = pd.to_datetime(df["date"])
 df["year"] = df["date"].dt.year
 df["month"] = df["date"].dt.month

 # === Agregaciones ===
 annual_means = {}
 monthly_means = {}

 for zone in ZONES.keys():
 df_zone = filter_by_zone(df, zone)

 annual_means[zone] = (
 df_zone.groupby("year")[variable].mean().reset_index()
 )
 monthly_means[zone] = (
 df_zone.groupby("month")[variable].mean().reset_index()
 )

 # === Visualización ===
 plt.figure(figsize=(8, 4))
 plt.plot(annual_means[zone]["year"], annual_means[zone][variable], marker="o")
 plt.title(f"Precipitación promedio anual – Zona {zone.capitalize()}")
 plt.xlabel("Año")
 plt.ylabel("Precipitación (m)")
 plt.grid(True)
 fig_path = FIG_DIR / f"era5_precip_{zone}_annual.png"
 plt.savefig(fig_path, dpi=150, bbox_inches="tight")
 plt.close()
 print(f" Figura guardada: {fig_path}")

 return annual_means, monthly_means

def run_full_pipeline():
 """
 Ejecuta todo el flujo: merge → diario → agregaciones → visualizaciones.
 """
 ds = merge_era5_files("total_precipitation")
 df_daily = compute_daily_means(ds, "total_precipitation")
 annual_means, monthly_means = compute_temporal_aggregates(df_daily)

 # Exporta resultados globales
 for zone, df_ in annual_means.items():
 df_.to_csv(OUT_DIR / f"era5_annual_{zone}.csv", index=False)
 for zone, df_ in monthly_means.items():
 df_.to_csv(OUT_DIR / f"era5_monthly_{zone}.csv", index=False)

 print("[OK] Flujo completo finalizado.")

if __name__ == "__main__":
 run_full_pipeline()

 """
 
 Cómo usarlo

Una vez descargados los archivos anuales con download_era5.py, ejecuta:

python src/utils/merge_era5_advanced.py

Esto generará automáticamente:

 En data/processed/

era5_precipitation_chile_full.nc

era5_precipitation_chile_daily.csv

era5_annual_norte.csv

era5_annual_centro.csv

era5_annual_sur.csv

era5_monthly_norte.csv

era5_monthly_centro.csv

era5_monthly_sur.csv

️ En reports/figures/

era5_precip_norte_annual.png

era5_precip_centro_annual.png

era5_precip_sur_annual.png

 Qué puedes hacer con estos resultados

En tu notebook 01_eda.ipynb, puedes cargar:

df_centro = pd.read_csv("data/processed/era5_annual_centro.csv")
plt.plot(df_centro["year"], df_centro["total_precipitation"])

Comparar comportamientos climáticos entre zonas.

Detectar tendencias o anomalías previas a modelado.

Alimentar tu Autoencoder con datos ya filtrados y balanceados espacialmente.

 Resumen del flujo completo
# 1. Descargar datos
python src/utils/download_era5.py

# 2. Combinar y procesar (avanzado)
python src/utils/merge_era5_advanced.py

# 3. Explorar
jupyter lab
# notebooks/01_eda.ipynb
 
 """