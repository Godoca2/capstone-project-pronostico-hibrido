"""fallback_build_from_npy.py
Crea archivos procesados a partir de `data/raw/precipitation_data.npy` para
desbloquear el EDA cuando aún no se han descargado los NetCDF de ERA5.

Genera:
 - data/processed/era5_precipitation_chile_full.nc
 - data/processed/era5_precipitation_chile_daily.csv

Asume que el array es diario con shape (T, 30). Reorganiza las 30 estaciones
en una grilla 6x5 sobre el bounding box de Chile, para obtener
`total_precipitation(time, latitude, longitude)` compatible con el notebook.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

def main():
 ROOT = Path(__file__).resolve().parent.parent.parent
 RAW = ROOT / "data" / "raw"
 OUT = ROOT / "data" / "processed"
 OUT.mkdir(parents=True, exist_ok=True)

 npy_path = RAW / "precipitation_data.npy"
 if not npy_path.exists():
 raise FileNotFoundError(f"No existe {npy_path}.")

 X = np.load(npy_path)
 if X.ndim != 2 or X.shape[1] != 30:
 raise ValueError(f"Se esperaba shape (T,30); recibido {X.shape}.")

 T, N = X.shape

 # Dimensiones espaciales sintéticas (6x5 = 30)
 nlat, nlon = 6, 5
 lats = np.linspace(-56, -17, nlat)
 lons = np.linspace(-76, -66, nlon)

 # Reordenar estaciones en grilla [time, lat, lon]
 X_grid = X.reshape(T, nlat, nlon)

 # Eje temporal diario a partir de 2020-01-01
 time_index = pd.date_range("2020-01-01", periods=T, freq="D")

 da = xr.DataArray(
 X_grid,
 dims=("time", "latitude", "longitude"),
 coords={"time": time_index, "latitude": lats, "longitude": lons},
 name="total_precipitation",
 )
 ds = da.to_dataset()

 out_nc = OUT / "era5_precipitation_chile_full.nc"
 ds.to_netcdf(out_nc)

 # CSV diario: ya es diario; exportamos por celda (lat, lon)
 df = da.to_dataframe().reset_index()
 df.rename(columns={"time": "date"}, inplace=True)
 out_csv = OUT / "era5_precipitation_chile_daily.csv"
 df.to_csv(out_csv, index=False)

 print("[OK] Fallback generado correctamente:")
 print(f" - {out_nc}")
 print(f" - {out_csv}")

if __name__ == "__main__":
 main()
