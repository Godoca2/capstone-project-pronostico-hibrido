"""data_check.py
=================
Script de validación rápida de integridad de datos del proyecto.

Verifica:
1. Existencia de archivos clave.
2. Shapes esperadas (matriz T×30, NetCDF con dims time/latitude/longitude).
3. Rango físico (precipitación >= 0).
4. Porcentaje de ceros y valores negativos.

Uso:
 python src/utils/data_check.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FILES = {
 "npy": RAW / "precipitation_data.npy",
 "csv_raw": RAW / "precipitation_test.csv",
 "stations_meta": RAW / "stations_meta.csv",
 "nc": PROC / "era5_precipitation_chile_full.nc",
 "csv_daily": PROC / "era5_precipitation_chile_daily.csv",
}

def check_exists():
 print("\n[EXISTENCIA]")
 for k, p in FILES.items():
 print(f" - {k}: {'OK' if p.exists() else 'NO'} -> {p}")

def check_npy_shape():
 p = FILES["npy"]
 if not p.exists():
 print("[NPY] Archivo ausente, skip.")
 return None
 arr = np.load(p)
 print(f"[NPY] shape={arr.shape}")
 if arr.ndim != 2 or arr.shape[1] != 30:
 print("[WARN] Se esperaba (T,30)")
 negatives = (arr < 0).sum()
 zeros = (arr == 0).sum()
 print(f"[NPY] negativos={negatives} zeros={zeros} pct_zeros={zeros/arr.size:.2%}")
 return arr

def check_csv_raw():
 p = FILES["csv_raw"]
 if not p.exists():
 print("[CSV_RAW] Ausente.")
 return None
 df = pd.read_csv(p)
 print(f"[CSV_RAW] shape={df.shape} columnas={list(df.columns)[:5]}...")
 if df.shape[1] != 30:
 print("[WARN] Se esperaban 30 columnas S1..S30")
 neg = (df.values < 0).sum()
 zeros = (df.values == 0).sum()
 print(f"[CSV_RAW] negativos={neg} zeros={zeros} pct_zeros={zeros/df.values.size:.2%}")
 return df

def check_nc():
 p = FILES["nc"]
 if not p.exists():
 print("[NC] Ausente.")
 return None
 try:
 ds = xr.open_dataset(p)
 except Exception as e:
 print(f"[NC] Error abriendo: {e}")
 return None
 print(f"[NC] dims={ds.dims} vars={list(ds.data_vars)}")
 needed = {"time", "latitude", "longitude"}
 if not needed.issubset(ds.dims):
 print("[WARN] Dims esperadas faltan")
 da = list(ds.data_vars.values())[0]
 neg = (da.values < 0).sum()
 zeros = (da.values == 0).sum()
 print(f"[NC] negativos={neg} zeros={zeros} pct_zeros={zeros/da.size:.2%}")
 return ds

def check_daily_csv():
 p = FILES["csv_daily"]
 if not p.exists():
 print("[CSV_DAILY] Ausente.")
 return None
 df = pd.read_csv(p)
 print(f"[CSV_DAILY] shape={df.shape} cols={list(df.columns)}")
 if not {"date", "latitude", "longitude", "total_precipitation"}.issubset(df.columns):
 print("[WARN] Columnas esperadas faltan")
 neg = (df["total_precipitation"] < 0).sum()
 zeros = (df["total_precipitation"] == 0).sum()
 print(f"[CSV_DAILY] negativos={neg} zeros={zeros} pct_zeros={zeros/len(df):.2%}")
 return df

def main():
 print("= DATA CHECK =")
 check_exists()
 check_npy_shape()
 check_csv_raw()
 check_nc()
 check_daily_csv()
 print("\n[OK] Validación básica completada.")

if __name__ == "__main__":
 main()
