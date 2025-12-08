"""train_ae_dmd_v2.py
=================================
Pipeline reproducible (script) para entrenar y evaluar un modelo híbrido
Autoencoder + Dynamic Mode Decomposition (AE-DMD) sobre series de
precipitación multiespacial, registrando todo en MLflow.

Resumen funcional
-----------------
1. Carga matriz de precipitación (real o sintética) desde `data/raw/`.
2. Divide en conjuntos de entrenamiento y prueba (split temporal 80/20).
3. Normaliza por media y desviación estándar del entrenamiento.
4. Ajusta el Autoencoder (con opción LSTM) y extrae representaciones latentes.
5. Ejecuta pronóstico usando dinámica DMD en espacio latente.
6. Denormaliza y calcula métricas (MAE, RMSE, NSE, R2, error reconstrucción).
7. Loggea parámetros, métricas y artefactos (gráfico, CSV, modelo) en MLflow.

Decisiones clave
----------------
* Se usa normalización simple (z-score) por estación para estabilizar el AE.
* El pronóstico se realiza con DMD sobre las latentes (dinámica lineal
 aproximada). `forecast(steps=len(X_test))` genera toda la secuencia futura.
* El modelo se serializa como pickle del autoencoder Keras por simplicidad.
* Se provee modo sintético para pruebas rápidas (`--use_synthetic`).

Uso rápido
----------
 python src/train_ae_dmd_v2.py --latent_dim 32 --epochs 50 --batch_size 32

Flags principales
-----------------
--latent_dim Dimensión del espacio latente del AE.
--use_lstm Activa arquitectura recurrente en lugar de densa pura.
--use_synthetic Reemplaza datos reales por serie sinusoidal + ruido (debug).
"""

import argparse
import mlflow
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend sin GUI
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.mlflow_utils import set_tracking
from models.ae_dmd import AE_DMD
from utils.metrics import evaluate_all

def load_real_data(data_path: str = "data/raw/precipitation_data.npy"):
 """Carga datos reales de precipitación desde un archivo `.npy`.

 El archivo contiene una matriz 2D donde cada columna representa una
 estación/celda y cada fila un instante temporal (día).

 Parameters
 ----------
 data_path : str
 Ruta relativa desde la raíz del proyecto al archivo `.npy`.

 Returns
 -------
 np.ndarray
 Array con shape (T, N_stations).
 """
 full_path = Path(__file__).parent.parent / data_path
 print(f"[DATA] Cargando datos desde: {full_path}")
 X = np.load(full_path)
 print(f"[DATA] Datos cargados: shape={X.shape} (T={X.shape[0]}, N_stations={X.shape[1]})")
 return X

def synthetic_data(T=365, N=50, seed=42):
 """Genera datos sintéticos (sinusoidal + ruido) para pruebas rápidas.

 Útil para validar pipeline sin depender de datos reales.
 """
 rng = np.random.default_rng(seed)
 base = np.sin(np.linspace(0, 6*np.pi, T)).reshape(-1, 1)
 noise = rng.normal(0, 0.1, size=(T, N))
 X = base @ np.linspace(0.5, 1.5, N).reshape(1, -1) + noise
 return X

def plot_forecast_comparison(y_true: np.ndarray, y_pred: np.ndarray, 
 station_idx: int = 0, save_path: str = None):
 """Genera gráfico comparando serie observada vs pronosticada.

 Parameters
 ----------
 y_true : np.ndarray
 Serie observada (test) con shape (T_test, N_stations).
 y_pred : np.ndarray
 Pronóstico generado por el modelo con misma forma.
 station_idx : int
 Índice de estación a visualizar.
 save_path : str | None
 Si se pasa, guarda el PNG como artefacto; en caso contrario muestra (no interactivo en backend Agg).
 """
 plt.figure(figsize=(12, 5))
 
 # Seleccionar una estación
 y_true_station = y_true[:, station_idx]
 y_pred_station = y_pred[:, station_idx]
 
 plt.plot(y_true_station, label='Observado', color='blue', linewidth=2)
 plt.plot(y_pred_station, label='Pronóstico AE-DMD', color='red', linestyle='--', linewidth=2)
 
 plt.xlabel('Tiempo (días)', fontsize=12)
 plt.ylabel('Precipitación', fontsize=12)
 plt.title(f'Pronóstico vs Observado - Estación {station_idx}', fontsize=14)
 plt.legend(fontsize=11)
 plt.grid(True, alpha=0.3)
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 print(f"[PLOT] Gráfico guardado: {save_path}")
 else:
 plt.show()
 
 plt.close()

def main(latent_dim: int, epochs: int, batch_size: int, use_lstm: bool = False, 
 use_synthetic: bool = False):
 """Ejecuta el pipeline completo de entrenamiento y evaluación AE-DMD.

 Parámetros se reciben desde CLI y se registran en MLflow.
 """
 # Configurar MLflow
 set_tracking()
 mlflow.set_experiment("AE_DMD_Precipitation_Forecast")

 with mlflow.start_run():
 # Log parámetros
 mlflow.log_param("latent_dim", latent_dim)
 mlflow.log_param("epochs", epochs)
 mlflow.log_param("batch_size", batch_size)
 mlflow.log_param("use_lstm", use_lstm)
 mlflow.log_param("use_synthetic", use_synthetic)

 # Cargar datos
 if use_synthetic:
 print("[DATA] Usando datos sintéticos")
 X = synthetic_data(T=365, N=80)
 else:
 X = load_real_data()

 # Split train/test
 T_train = int(0.8 * len(X))
 X_train, X_test = X[:T_train], X[T_train:]
 print(f"[SPLIT] Train: {X_train.shape}, Test: {X_test.shape}")

 # Normalización (z-score por estación)
 mean_train = X_train.mean(axis=0)
 std_train = X_train.std(axis=0) + 1e-8
 X_train_norm = (X_train - mean_train) / std_train
 X_test_norm = (X_test - mean_train) / std_train

 # Entrenar modelo AE-DMD (AE para compresión + DMD para dinámica)
 print("\n" + "="*60)
 print("ENTRENAMIENTO AE-DMD")
 print("="*60)
 
 model = AE_DMD(latent_dim=latent_dim, use_lstm=use_lstm)
 model.fit(X_train_norm, X_time_series=X_train_norm, 
 epochs=epochs, batch_size=batch_size)

 # Forecasting en espacio latente (método 'dmd')
 print("\n" + "="*60)
 print("FORECASTING")
 print("="*60)
 
 y_pred_norm = model.forecast(steps=len(X_test), method='dmd')
 
 # Denormalizar pronóstico para regresar a unidades originales
 y_pred = y_pred_norm * std_train + mean_train
 y_true = X_test

 # Evaluar métricas globales (flatten para considerar todas las estaciones)
 print("\n" + "="*60)
 print("EVALUACIÓN")
 print("="*60)
 
 metrics = evaluate_all(y_true.flatten(), y_pred.flatten())
 
 for metric_name, metric_value in metrics.items():
 print(f"{metric_name}: {metric_value:.4f}")
 mlflow.log_metric(metric_name, metric_value)

 # Error de reconstrucción del AE (indicador de calidad de codificación)
 reconstruction_error = model.get_reconstruction_error(X_train_norm)
 print(f"Reconstruction Error (AE): {reconstruction_error:.6f}")
 mlflow.log_metric("reconstruction_error", reconstruction_error)

 # Guardar artefactos (gráfico, CSV de un ejemplo y modelo serializado)
 reports_path = Path(__file__).parent.parent / "reports" / "figures"
 reports_path.mkdir(parents=True, exist_ok=True)
 
 # 1. Gráfico comparativo
 plot_path = reports_path / "ae_dmd_forecast_comparison.png"
 plot_forecast_comparison(y_true, y_pred, station_idx=0, save_path=str(plot_path))
 mlflow.log_artifact(str(plot_path))
 
 # 2. Predicciones CSV
 pred_csv_path = reports_path / "predictions.csv"
 np.savetxt(pred_csv_path, 
 np.c_[y_true[:, 0], y_pred[:, 0]], 
 delimiter=",", 
 header="observed,predicted", 
 comments="")
 mlflow.log_artifact(str(pred_csv_path))
 
 # 3. Guardar modelo Keras usando pickle
 import pickle
 model_path = reports_path / "autoencoder_model.pkl"
 with open(model_path, 'wb') as f:
 pickle.dump(model.autoencoder, f)
 mlflow.log_artifact(str(model_path))
 
 print("\n" + "="*60)
 print(f"[OK] Experimento completado exitosamente")
 print(f" MAE: {metrics['MAE']:.4f}")
 print(f" RMSE: {metrics['RMSE']:.4f}")
 print(f" NSE: {metrics['NSE']:.4f}")
 print(f" R2: {metrics['R2']:.4f}")
 print("="*60)

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="Entrenamiento AE-DMD con MLflow")
 parser.add_argument("--latent_dim", type=int, default=32, 
 help="Dimensión del espacio latente (default: 32)")
 parser.add_argument("--epochs", type=int, default=50, 
 help="Número de épocas de entrenamiento (default: 50)")
 parser.add_argument("--batch_size", type=int, default=32, 
 help="Tamaño de batch (default: 32)")
 parser.add_argument("--use_lstm", action="store_true", 
 help="Usar arquitectura LSTM en lugar de densa")
 parser.add_argument("--use_synthetic", action="store_true", 
 help="Usar datos sintéticos en lugar de reales")
 
 args = parser.parse_args()
 main(**vars(args))
