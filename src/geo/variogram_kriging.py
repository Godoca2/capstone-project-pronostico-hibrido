"""variogram_kriging.py
=======================
Cálculo de variograma experimental y kriging ordinario para un snapshot
(diario o mensual) de precipitación.

Compatibilidad fuentes:
- CSV diario procesado: data/processed/era5_precipitation_chile_daily.csv
- Si no existe, intenta reconstruir puntos desde NetCDF combinado.

Dependencias: scikit-gstat, pykrige, numpy, pandas, xarray, matplotlib

Uso:
 python src/geo/variogram_kriging.py --date 2020-01-15
 python src/geo/variogram_kriging.py --month 2020-01
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paquetes geoestadísticos
try:
 from skgstat import Variogram
except Exception as e:
 raise SystemExit("[ERROR] scikit-gstat no instalado. pip install scikit-gstat") from e

try:
 from pykrige.ok import OrdinaryKriging
except Exception as e:
 raise SystemExit("[ERROR] PyKrige no instalado. pip install pykrige") from e

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
OUT = PROC
FIG.mkdir(parents=True, exist_ok=True)

def load_points_snapshot(date: str | None = None, month: str | None = None) -> pd.DataFrame:
 """Carga puntos (lat, lon, value) para una fecha o mes.

 Si `month` está definido (YYYY-MM), agrega por suma mensual.
 """
 csv = PROC / "era5_precipitation_chile_daily.csv"
 if csv.exists():
 df = pd.read_csv(csv, parse_dates=["date"]) # columns: date, latitude, longitude, total_precipitation
 else:
 nc = PROC / "era5_precipitation_chile_full.nc"
 if not nc.exists():
 raise FileNotFoundError("No se encontró dataset procesado. Ejecuta merge_era5_small.py")
 ds = xr.open_dataset(nc)
 da = list(ds.data_vars.values())[0]
 df = da.to_dataframe().reset_index().rename(columns={"time": "date", da.name: "total_precipitation"})

 if month:
 # Sumar por mes
 df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M")
 dfm = (
 df[df["ym"] == month]
 .groupby(["latitude", "longitude"], as_index=False)["total_precipitation"].sum()
 )
 dfm.rename(columns={"total_precipitation": "value"}, inplace=True)
 return dfm[["latitude", "longitude", "value"]]

 if date:
 dfd = df[pd.to_datetime(df["date"]).dt.date == pd.to_datetime(date).date()].copy()
 dfd.rename(columns={"total_precipitation": "value"}, inplace=True)
 return dfd[["latitude", "longitude", "value"]]

 raise ValueError("Debe especificar --date o --month")

def compute_variogram(df_points: pd.DataFrame, model: str = "spherical") -> Variogram:
 coords = df_points[["latitude", "longitude"]].to_numpy()
 values = df_points["value"].to_numpy()
 V = Variogram(coords, values, model=model, normalize=False, use_nugget=True, maxlag="median")
 return V

def plot_variogram(V: Variogram, title: str, path: Path):
 fig, ax = plt.subplots(figsize=(6, 4))
 V.plot(axes=ax)
 ax.set_title(title)
 fig.tight_layout()
 fig.savefig(path, dpi=150)
 plt.close(fig)

essENTIAL_EPS = 1e-6

def run_kriging(df_points: pd.DataFrame, grid_res: float = 0.25):
 # PyKrige espera x,y → usaremos lon=x, lat=y
 xs = df_points["longitude"].to_numpy()
 ys = df_points["latitude"].to_numpy()
 zs = df_points["value"].to_numpy()

 # Evitar exactamente cero varianza
 if np.allclose(zs.std(), 0.0):
 zs = zs + essENTIAL_EPS

 OK = OrdinaryKriging(xs, ys, zs, variogram_model="spherical", verbose=False, enable_plotting=False)

 lon_min, lon_max = xs.min(), xs.max()
 lat_min, lat_max = ys.min(), ys.max()

 grid_lon = np.arange(lon_min, lon_max + grid_res, grid_res)
 grid_lat = np.arange(lat_min, lat_max + grid_res, grid_res)

 zhat, zvar = OK.execute("grid", grid_lon, grid_lat)
 return grid_lon, grid_lat, zhat, zvar

def plot_kriging(grid_lon, grid_lat, zhat, title: str, path: Path):
 fig, ax = plt.subplots(figsize=(7, 5))
 im = ax.imshow(
 zhat,
 origin="lower",
 extent=[grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()],
 cmap="Blues",
 aspect="auto",
 )
 fig.colorbar(im, ax=ax, label="precip")
 ax.set_xlabel("lon")
 ax.set_ylabel("lat")
 ax.set_title(title)
 fig.tight_layout()
 fig.savefig(path, dpi=150)
 plt.close(fig)

def save_nc(grid_lon, grid_lat, zhat, path: Path, date_label: str):
 da = xr.DataArray(
 np.array(zhat),
 dims=("latitude", "longitude"),
 coords={"latitude": grid_lat, "longitude": grid_lon},
 name="precip_kriged",
 )
 da = da.expand_dims({"time": [np.datetime64(date_label)]})
 ds = da.to_dataset()
 ds.to_netcdf(path)

def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--date", type=str, default=None, help="Fecha YYYY-MM-DD a analizar")
 parser.add_argument("--month", type=str, default=None, help="Mes YYYY-MM a agregar por suma")
 parser.add_argument("--grid_res", type=float, default=0.25)
 args = parser.parse_args()

 dfp = load_points_snapshot(date=args.date, month=args.month)
 label = args.date or args.month

 # Variograma
 V = compute_variogram(dfp)
 var_path = FIG / f"variogram_{label}.png"
 plot_variogram(V, f"Variograma {label}", var_path)
 print(f"[SAVE] {var_path}")

 # Kriging
 glon, glat, zhat, zvar = run_kriging(dfp, grid_res=args.grid_res)
 krig_fig = FIG / f"kriging_map_{label}.png"
 plot_kriging(glon, glat, zhat, f"Kriging {label}", krig_fig)
 print(f"[SAVE] {krig_fig}")

 out_nc = OUT / f"kriging_{label}.nc"
 save_nc(glon, glat, zhat, out_nc, date_label=label + "-01" if args.month else label)
 print(f"[SAVE] {out_nc}")

if __name__ == "__main__":
 main()
