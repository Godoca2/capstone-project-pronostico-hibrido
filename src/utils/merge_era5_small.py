"""merge_era5_small.py
Combina los .nc mensuales de ERA5 (total_precipitation, 2020) en un solo
dataset y genera:
 - data/processed/era5_precipitation_chile_full.nc
 - data/processed/era5_precipitation_chile_daily.csv

Uso:
 conda activate capstone
 python src/utils/merge_era5_small.py
"""
from pathlib import Path
import pandas as pd
import xarray as xr

def _open_robusto(files: list[Path]) -> xr.Dataset:
 """Intenta abrir con open_mfdataset; si falla, concatena manualmente."""
 try:
 ds = xr.open_mfdataset(
 [str(f) for f in files],
 combine="by_coords",
 data_vars="minimal",
 coords="minimal",
 compat="override",
 parallel=False,
 )
 return ds
 except Exception as e:
 print(f"[WARN] open_mfdataset falló: {e}. Intentando concatenación manual…")
 datasets = [xr.open_dataset(str(f)) for f in files]
 ds = xr.concat(datasets, dim="time")
 return ds

def main():
 ROOT = Path(__file__).resolve().parent.parent.parent
 RAW = ROOT / "data" / "raw"
 OUT = ROOT / "data" / "processed"
 OUT.mkdir(parents=True, exist_ok=True)

 files = sorted(RAW.glob("era5_precipitation_chile_2020_*.nc"))
 if not files:
 raise FileNotFoundError("No se encontraron archivos en data/raw/ con patrón era5_precipitation_chile_2020_*.nc")

 print(f"[INFO] Encontrados {len(files)} archivos para combinar.")
 ds = _open_robusto(files)

 # Asegurar variable principal
 var = "total_precipitation"
 if var not in ds.data_vars:
 # Si el nombre difiere, tomar la primera
 var = list(ds.data_vars)[0]
 print(f"[WARN] Variable '{var}' seleccionada automáticamente.")

 # Identificar dimensión temporal
 time_dim = None
 for dim_name in ["time", "valid_time", "datetime"]:
 if dim_name in ds.dims:
 time_dim = dim_name
 break
 
 if not time_dim:
 raise ValueError(f"No se encontró dimensión temporal. Dimensiones disponibles: {list(ds.dims)}")
 
 print(f"[INFO] Usando dimensión temporal: '{time_dim}'")
 
 # Ordenar
 order_dims = [d for d in [time_dim, "latitude", "longitude"] if d in ds.dims]
 if order_dims:
 ds = ds.sortby(order_dims)

 # Guardar NetCDF combinado
 out_nc = OUT / "era5_precipitation_chile_full.nc"
 print(f"[SAVE] NetCDF combinado → {out_nc}")
 ds[[var]].to_netcdf(out_nc)

 # Agregado diario: sumatorio por día
 print("[INFO] Resample diario (sum)…")
 da_daily = ds[var].resample({time_dim: "1D"}).sum()
 df_daily = da_daily.to_dataframe().reset_index()

 out_csv = OUT / "era5_precipitation_chile_daily.csv"
 print(f"[SAVE] CSV diario → {out_csv}")
 df_daily.to_csv(out_csv, index=False)

 print("\n[OK] Merge finalizado.")
 print(f" - {out_nc}")
 print(f" - {out_csv}")

if __name__ == "__main__":
 main()
