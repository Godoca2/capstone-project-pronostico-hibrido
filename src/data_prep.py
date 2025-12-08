"""
data_prep.py
-------------
Módulo para lectura, exploración y procesamiento inicial de datos climáticos
en formato NetCDF (ej. ERA5). Convierte los datos a pandas.DataFrame y deja
estructuras listas para modelamiento y análisis.

Autor: César Godoy Delaigue
Magíster Data Science UDD - 2025
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

def load_era5_dataset(netcdf_path: str, variable: str = "precipitation") -> xr.Dataset:
 """
 Carga un archivo NetCDF (ej. ERA5) y retorna un xarray.Dataset.
 
 Parámetros
 ----------
 netcdf_path : str
 Ruta al archivo .nc
 variable : str
 Nombre de la variable a analizar (e.g. "tp", "precipitation")
 
 Retorna
 -------
 ds : xr.Dataset
 Dataset de ERA5 con coordenadas (time, latitude, longitude)
 """
 print(f"Cargando archivo NetCDF: {netcdf_path}")
 ds = xr.open_dataset(netcdf_path)
 print("Variables disponibles:", list(ds.data_vars))
 if variable not in ds.data_vars:
 print(f"[AVISO] Variable '{variable}' no encontrada. Se usará la primera disponible.")
 variable = list(ds.data_vars)[0]
 ds = ds[[variable]]
 ds = ds.sortby(["latitude", "longitude", "time"])
 return ds

def to_dataframe(ds: xr.Dataset, variable: str = None) -> pd.DataFrame:
 """
 Convierte un xarray.Dataset a pandas.DataFrame plano con columnas:
 ['time', 'latitude', 'longitude', 'value'].
 """
 if variable is None:
 variable = list(ds.data_vars)[0]
 df = ds[variable].to_dataframe().reset_index()
 df.rename(columns={variable: "value"}, inplace=True)
 print(f"DataFrame generado: {df.shape[0]:,} registros")
 return df

def subset_region(df: pd.DataFrame, lat_range: tuple, lon_range: tuple) -> pd.DataFrame:
 """
 Filtra un DataFrame por rango espacial.
 """
 return df[
 (df["latitude"].between(*lat_range)) &
 (df["longitude"].between(*lon_range))
 ]

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
 """
 Agrega valores horarios o subdiarios a frecuencia diaria (media o suma).
 """
 df["date"] = pd.to_datetime(df["time"]).dt.date
 daily = df.groupby(["date", "latitude", "longitude"])["value"].sum().reset_index()
 return daily

def save_processed(df: pd.DataFrame, out_path: str = "data/processed/era5_daily.csv"):
 """
 Guarda el DataFrame procesado como CSV.
 """
 Path(out_path).parent.mkdir(parents=True, exist_ok=True)
 df.to_csv(out_path, index=False)
 print(f"[OK] Archivo guardado: {out_path}")

# Ejemplo de uso rápido (ejecutar manualmente si se corre este archivo)
if __name__ == "__main__":
 path = "data/raw/era5_precipitation_sample.nc"
 ds = load_era5_dataset(path, "tp")
 df = to_dataframe(ds, "tp")
 df_cl = subset_region(df, lat_range=(-56, -17), lon_range=(-75, -66))
 df_daily = aggregate_daily(df_cl)
 save_processed(df_daily)
