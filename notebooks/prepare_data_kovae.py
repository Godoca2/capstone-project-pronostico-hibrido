"""prepare_data_kovae.py
=====================
Script para preparar datos ERA5 2020 completos para entrenamiento KoVAE.

Carga el dataset ERA5 2020 horario y lo agrega a diario (366 días) 
para entrenamiento KoVAE.

Autor: Capstone Project
Fecha: 19 Nov 2025
"""

import numpy as np
import xarray as xr
import pickle
from pathlib import Path

# Directorios
DATA_DIR = Path('../data')
ERA5_PATH = DATA_DIR / 'processed' / 'era5_precipitation_chile_full.nc'
OUTPUT_PATH = DATA_DIR / 'processed' / 'era5_2020_daily_for_kovae.pkl'

print("📦 Cargando datos ERA5 2020 (horario)...")
ds = xr.open_dataset(ERA5_PATH)

# Verificar estructura
print(f"\n📊 Dataset ERA5:")
print(f"   Variables: {list(ds.data_vars)}")
print(f"   Coordenadas: {list(ds.coords)}")
print(f"   Shape: {ds['tp'].shape if 'tp' in ds else ds['precipitation'].shape if 'precipitation' in ds else 'N/A'}")

# Identificar variable de precipitación
precip_var = 'tp' if 'tp' in ds else 'precipitation' if 'precipitation' in ds else None

if precip_var is None:
    raise ValueError(f"No se encontró variable de precipitación. Variables disponibles: {list(ds.data_vars)}")

# Identificar dimensión temporal
time_dim = 'valid_time' if 'valid_time' in ds else 'time'

print(f"\n⏰ Dimensión temporal: {time_dim}")
print(f"   Periodo: {ds[time_dim].values[0]} a {ds[time_dim].values[-1]}")
print(f"   Registros: {len(ds[time_dim])} (horario)")

# Agregar a datos diarios (suma cada 24 horas)
print(f"\n🔄 Agregando datos horarios a diarios...")
ds_daily = ds.resample({time_dim: '1D'}).sum()

# Agregar a datos diarios (suma cada 24 horas)
print(f"\n🔄 Agregando datos horarios a diarios...")
ds_daily = ds.resample({time_dim: '1D'}).sum()

print(f"   ✅ Agregación completada: {len(ds_daily[time_dim])} días")

print(f"\n✅ Datos 2020 diarios:")
print(f"   Shape: {ds_daily[precip_var].shape}")
print(f"   Periodo: {ds_daily[time_dim].values[0]} a {ds_daily[time_dim].values[-1]}")

# Convertir a numpy array con shape (days, lat, lon, 1)
precip_data = ds_daily[precip_var].values

# Asegurar 4 dimensiones (days, lat, lon, channels)
if precip_data.ndim == 3:
    precip_data = np.expand_dims(precip_data, axis=-1)

print(f"\n📐 Array final:")
print(f"   Shape: {precip_data.shape}")
print(f"   Rango: [{precip_data.min():.4f}, {precip_data.max():.4f}]")
print(f"   Mean: {precip_data.mean():.4f}")
print(f"   Std: {precip_data.std():.4f}")

# Guardar
data_dict = {
    'precip_2020': precip_data,
    'time': ds_daily[time_dim].values,
    'lat': ds_daily.latitude.values if 'latitude' in ds_daily else ds_daily.lat.values,
    'lon': ds_daily.longitude.values if 'longitude' in ds_daily else ds_daily.lon.values,
    'spatial_dims': precip_data.shape[1:3]
}

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"\n💾 Datos guardados en: {OUTPUT_PATH}")
print(f"   Tamaño: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")
print(f"\n✅ Preparación completada")
