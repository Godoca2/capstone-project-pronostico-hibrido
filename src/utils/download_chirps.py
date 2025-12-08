"""download_chirps.py
===================
Script para descargar datos CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
de precipitación satelital para Chile.

CHIRPS proporciona datos de precipitación diaria desde 1981 hasta casi tiempo real.
Resolución espacial: 0.05° (~5.5 km en el ecuador)

Fuentes de datos:
- CHIRPS v2.0: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
- FTP: ftp://ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/
- API Climate Engine: https://climateengine.org/

Autor: Capstone Project - Pronóstico Híbrido Precipitaciones Chile
Fecha: 19 Nov 2025
"""

import os
import requests
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def download_chirps_daily(
 start_date: str,
 end_date: str,
 lat_min: float = -56.0,
 lat_max: float = -17.0,
 lon_min: float = -76.0,
 lon_max: float = -66.0,
 output_dir: str = '../../data/external/chirps',
 force_download: bool = False
):
 """
 Descarga datos CHIRPS diarios para una región y periodo específico.
 
 Parameters
 ----------
 start_date : str
 Fecha inicial en formato 'YYYY-MM-DD'
 end_date : str
 Fecha final en formato 'YYYY-MM-DD'
 lat_min, lat_max : float
 Rango de latitudes (Chile: -56° a -17°)
 lon_min, lon_max : float
 Rango de longitudes (Chile: -76° a -66°)
 output_dir : str
 Directorio donde guardar los archivos descargados
 force_download : bool
 Si True, reemplaza archivos existentes
 
 Returns
 -------
 xr.Dataset
 Dataset con precipitación CHIRPS para el periodo solicitado
 
 Notes
 -----
 CHIRPS URL pattern: 
 https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/
 chirps-v2.0.YYYY.days_p05.nc (archivo anual)
 
 Para descargas grandes, considerar usar FTP o Climate Engine API.
 """
 
 # Crear directorio de salida
 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)
 
 # Parsear fechas
 start = datetime.strptime(start_date, '%Y-%m-%d')
 end = datetime.strptime(end_date, '%Y-%m-%d')
 
 # Determinar años a descargar
 years = list(range(start.year, end.year + 1))
 
 print(f"[INFO] Descargando datos CHIRPS para Chile")
 print(f" Periodo: {start_date} a {end_date}")
 print(f" Región: Lat[{lat_min}, {lat_max}], Lon[{lon_min}, {lon_max}]")
 print(f" Años: {years}")
 
 datasets = []
 
 for year in years:
 # Archivo local
 filename = f"chirps-v2.0.{year}.days_p05.nc"
 local_path = output_path / filename
 
 # URL de descarga
 base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
 url = f"{base_url}{filename}"
 
 # Descargar si no existe o force_download=True
 if not local_path.exists() or force_download:
 print(f"\n⏬ Descargando {year}...")
 try:
 response = requests.get(url, stream=True)
 response.raise_for_status()
 
 total_size = int(response.headers.get('content-length', 0))
 
 with open(local_path, 'wb') as f, tqdm(
 total=total_size,
 unit='B',
 unit_scale=True,
 unit_divisor=1024,
 desc=f"CHIRPS {year}"
 ) as pbar:
 for chunk in response.iter_content(chunk_size=8192):
 f.write(chunk)
 pbar.update(len(chunk))
 
 print(f" [OK] Guardado: {local_path}")
 
 except requests.exceptions.RequestException as e:
 print(f" [ERROR] Error descargando {year}: {e}")
 continue
 else:
 print(f" Archivo existente: {filename}")
 
 # Cargar y recortar región de Chile
 try:
 ds = xr.open_dataset(local_path)
 
 # CHIRPS usa 'latitude', 'longitude', 'time', variable 'precip'
 # Recortar región
 ds_chile = ds.sel(
 latitude=slice(lat_max, lat_min), # xarray slice invierte orden
 longitude=slice(lon_min, lon_max)
 )
 
 # Filtrar fechas del periodo solicitado
 if year == start.year:
 ds_chile = ds_chile.sel(time=slice(start_date, f"{year}-12-31"))
 elif year == end.year:
 ds_chile = ds_chile.sel(time=slice(f"{year}-01-01", end_date))
 
 datasets.append(ds_chile)
 
 except Exception as e:
 print(f" [ERROR] Error cargando {filename}: {e}")
 continue
 
 # Concatenar datasets de todos los años
 if datasets:
 print(f"\n[INFO] Concatenando {len(datasets)} dataset(s)...")
 ds_combined = xr.concat(datasets, dim='time')
 
 # Guardar dataset combinado
 output_combined = output_path / f"chirps_chile_{start_date}_{end_date}.nc"
 print(f"[Guardado] Guardando dataset combinado: {output_combined}")
 ds_combined.to_netcdf(output_combined)
 
 print(f"\n[OK] Descarga completada:")
 print(f" Shape: {ds_combined['precip'].shape}")
 print(f" Variables: {list(ds_combined.data_vars)}")
 print(f" Coordenadas: {list(ds_combined.coords)}")
 print(f" Archivo: {output_combined}")
 
 return ds_combined
 
 else:
 raise ValueError("No se pudieron descargar datos CHIRPS")

def compare_with_era5(
 chirps_path: str,
 era5_path: str,
 output_dir: str = '../../reports/figures'
):
 """
 Compara datos CHIRPS (satelital) con ERA5 (reanálisis).
 
 Parameters
 ----------
 chirps_path : str
 Ruta al archivo NetCDF con datos CHIRPS
 era5_path : str
 Ruta al archivo NetCDF con datos ERA5
 output_dir : str
 Directorio para guardar figuras de comparación
 
 Notes
 -----
 CHIRPS: Resolución 0.05° (~5.5 km), datos satelitales + estaciones
 ERA5: Resolución 0.25° (~27.8 km), modelo de reanálisis
 
 Para comparar, necesitamos interpolar a la misma resolución.
 """
 
 import matplotlib.pyplot as plt
 import seaborn as sns
 from scipy.interpolate import griddata
 
 print(f"[INFO] Comparando CHIRPS vs ERA5...")
 
 # Cargar datasets
 ds_chirps = xr.open_dataset(chirps_path)
 ds_era5 = xr.open_dataset(era5_path)
 
 print(f"\n[INFO] Shapes:")
 print(f" CHIRPS: {ds_chirps['precip'].shape}")
 print(f" ERA5: {ds_era5['precipitation'].shape if 'precipitation' in ds_era5 else ds_era5['tp'].shape}")
 
 # Implementar comparación detallada aquí
 # TODO: Interpolar a resolución común, calcular correlaciones espaciales,
 # generar mapas de diferencias, scatter plots, series temporales
 
 print(f"\n[AVISO] Comparación detallada pendiente de implementación")
 print(f" Sugerencias:")
 print(f" - Interpolar CHIRPS a resolución ERA5 (0.25°)")
 print(f" - Calcular correlación espacial por pixel")
 print(f" - Comparar estadísticas (media, varianza, percentiles)")
 print(f" - Generar mapas de bias y RMSE")

if __name__ == '__main__':
 """
 Ejemplo de uso:
 python download_chirps.py
 """
 
 # Configuración para Chile
 START_DATE = '2019-01-01'
 END_DATE = '2020-02-29' # Periodo que coincide con datos ERA5 del proyecto
 
 # Bounding box Chile (mismo que ERA5)
 LAT_MIN = -56.0
 LAT_MAX = -17.0
 LON_MIN = -76.0
 LON_MAX = -66.0
 
 # Descargar datos CHIRPS
 try:
 ds_chirps = download_chirps_daily(
 start_date=START_DATE,
 end_date=END_DATE,
 lat_min=LAT_MIN,
 lat_max=LAT_MAX,
 lon_min=LON_MIN,
 lon_max=LON_MAX,
 output_dir='../../data/external/chirps',
 force_download=False
 )
 
 print(f"\n Proceso completado exitosamente")
 
 except Exception as e:
 print(f"\n[ERROR] Error en la descarga: {e}")
 print(f"\n[NOTA] Alternativas:")
 print(f" 1. Usar FTP directo: ftp://ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/")
 print(f" 2. Climate Engine API: https://climateengine.org/")
 print(f" 3. Google Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY")
