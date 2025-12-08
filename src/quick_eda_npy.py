"""quick_eda_npy.py
Genera un EDA rápido a partir de `data/raw/precipitation_data.npy`.

Outputs:
 - `reports/eda_summary.csv` (estadísticas por tiempo y por estación)
 - `reports/figures/mean_time_series.png`
 - `reports/figures/station_variance.png`

Usar: python src/quick_eda_npy.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def main():
 p = RAW / "precipitation_data.npy"
 if not p.exists():
 raise FileNotFoundError(f"Archivo no encontrado: {p}")

 X = np.load(p)
 # Esperamos shape (T, N)
 T, N = X.shape

 # Estadísticas temporales (promedio sobre estaciones)
 mean_time = X.mean(axis=1)
 std_time = X.std(axis=1)

 # Estadísticas por estación (promedio sobre tiempo)
 mean_station = X.mean(axis=0)
 var_station = X.var(axis=0)

 # Guardar resumen CSV
 df_time = pd.DataFrame({"day_index": range(T), "mean_precip": mean_time, "std_precip": std_time})
 df_station = pd.DataFrame({"station_index": range(N), "mean_precip": mean_station, "var_precip": var_station})

 summary_time_path = REPORTS / "eda_time_summary.csv"
 summary_station_path = REPORTS / "eda_station_summary.csv"
 df_time.to_csv(summary_time_path, index=False)
 df_station.to_csv(summary_station_path, index=False)

 # Plot time series mean
 plt.figure(figsize=(10,4))
 sns.lineplot(x=df_time['day_index'], y=df_time['mean_precip'])
 plt.xlabel('Day index')
 plt.ylabel('Mean precipitation (all stations)')
 plt.title('Mean precipitation over time')
 out1 = FIGS / 'mean_time_series.png'
 plt.tight_layout()
 plt.savefig(out1, dpi=150)
 plt.close()

 # Plot station variance (top 30)
 topk = min(30, N)
 df_station_sorted = df_station.sort_values('var_precip', ascending=False).reset_index(drop=True).head(topk)
 plt.figure(figsize=(10,4))
 sns.barplot(x='station_index', y='var_precip', data=df_station_sorted)
 plt.xlabel('Station index (top variability)')
 plt.ylabel('Variance of precipitation')
 plt.title(f'Top {topk} stations by variance')
 out2 = FIGS / 'station_variance.png'
 plt.tight_layout()
 plt.savefig(out2, dpi=150)
 plt.close()

 print(f"[OK] EDA rápido completado. Archivos generados:")
 print(f" - {summary_time_path}")
 print(f" - {summary_station_path}")
 print(f" - {out1}")
 print(f" - {out2}")

if __name__ == '__main__':
 main()
