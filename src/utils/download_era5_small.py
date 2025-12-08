"""download_era5_small.py
Descarga ERA5 (total_precipitation) únicamente para 2020 y para el bounding box de Chile,
en archivos mensuales .nc dentro de data/raw/.

Requisitos:
- Cuenta CDS y archivo de credenciales en ~/.cdsapirc
- Paquetes: cdsapi (pip install cdsapi)

Uso:
 conda activate capstone
 python src/utils/download_era5_small.py

Esto creará hasta 12 archivos en data/raw/:
 era5_precipitation_chile_2020_01.nc ... era5_precipitation_chile_2020_12.nc
"""
from pathlib import Path
import sys

def ensure_cdsapi():
 try:
 import cdsapi # noqa: F401
 except Exception:
 print("[ERROR] No se encontró 'cdsapi'. Instálalo con: pip install cdsapi")
 sys.exit(1)

def ensure_cds_config():
 cfg = Path.home() / ".cdsapirc"
 if not cfg.exists():
 print("[ERROR] No se encontró el archivo de credenciales ~/.cdsapirc")
 print(" Crea una cuenta en https://cds.climate.copernicus.eu/ y configura tus credenciales.")
 sys.exit(1)

def main():
 ensure_cdsapi()

 import cdsapi

 # Rutas
 ROOT = Path(__file__).resolve().parent.parent.parent
 RAW = ROOT / "data" / "raw"
 RAW.mkdir(parents=True, exist_ok=True)

 # Parámetros de área (Chile) en formato [N, W, S, E]
 area = [ -17, -76, -56, -66 ]
 year = "2020"
 months = [f"{m:02d}" for m in range(1, 13)]

 # Horas y días (hora en UTC)
 hours = [f"{h:02d}:00" for h in range(24)]
 days = [f"{d:02d}" for d in range(1, 32)] # el API ignora días inexistentes por mes

 # Cliente con credenciales explícitas
 c = cdsapi.Client(
 url="https://cds.climate.copernicus.eu/api",
 key="db6d5c46-cd6e-40e2-87fd-1954350a3cbe",
 verify=True
 )

 for mm in months:
 out_nc = RAW / f"era5_precipitation_chile_{year}_{mm}.nc"
 if out_nc.exists():
 print(f"[SKIP] Ya existe {out_nc}")
 continue

 print(f"[DL] Descargando ERA5 {year}-{mm} → {out_nc}")
 c.retrieve(
 "reanalysis-era5-single-levels",
 {
 "product_type": "reanalysis",
 "format": "netcdf",
 "variable": ["total_precipitation"],
 "year": year,
 "month": mm,
 "day": days,
 "time": hours,
 "area": area, # [North, West, South, East]
 "grid": [0.25, 0.25],
 },
 str(out_nc),
 )

 print("\n[OK] Descarga finalizada. Archivos en:")
 print(RAW)

if __name__ == "__main__":
 main()
