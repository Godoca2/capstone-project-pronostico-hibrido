import xarray as xr
from pathlib import Path

ERA5_PATH = Path('../data/processed/era5_precipitation_chile_full.nc')
ds = xr.open_dataset(ERA5_PATH)

time_dim = 'valid_time' if 'valid_time' in ds else 'time'

print(f"Dimensión temporal: {time_dim}")
print(f"Primeros 10 timestamps:")
print(ds[time_dim].values[:10])
print(f"\nÚltimos 10 timestamps:")
print(ds[time_dim].values[-10:])
print(f"\nTotal: {len(ds[time_dim])} registros")
