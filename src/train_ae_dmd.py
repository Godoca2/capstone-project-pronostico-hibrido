"""Este script ya deja todo conectado a MLflow: parámetros, métricas y artefactos. 
Una vez valides que corre, reemplazas el generador sintético por 
tu pipeline real (ERA5/CHIRPS → variogramas/kriging → AE → DMD)."""

import argparse
import mlflow
import mlflow.sklearn
import numpy as np
from utils.mlflow_utils import set_tracking
from models.ae_dmd import SimpleAEdmd

def synthetic_data(T=365, N=50, seed=42):
 # Datos sintéticos para probar el pipeline (ej: “precipitacion” simulada)
 rng = np.random.default_rng(seed)
 base = np.sin(np.linspace(0, 6*np.pi, T)).reshape(-1,1)
 noise = rng.normal(0, 0.1, size=(T, N))
 X = base @ np.linspace(0.5, 1.5, N).reshape(1,-1) + noise
 return X

def main(latent_dim, epochs, batch_size):
 set_tracking()
 mlflow.set_experiment("AE_DMD_basico")

 with mlflow.start_run():
 # Log params
 mlflow.log_param("latent_dim", latent_dim)
 mlflow.log_param("epochs", epochs)
 mlflow.log_param("batch_size", batch_size)

 # “Carga” de datos (aquí sintético; luego cambia a ERA5/CHIRPS procesado)
 X = synthetic_data(T=365, N=80)
 T_train = 300
 X_train, X_test = X[:T_train], X[T_train:]

 model = SimpleAEdmd(latent_dim=latent_dim).fit(X_train)

 # “Pronóstico” de 65 pasos para comparar con test
 y_pred = model.forecast(steps=len(X_test))
 y_true = X_test

 # Métricas simples
 mae = float(np.mean(np.abs(y_true - y_pred)))
 rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))

 mlflow.log_metric("MAE", mae)
 mlflow.log_metric("RMSE", rmse)

 # Guarda artefactos ligeros (predicciones)
 np.savetxt("reports/figures/pred_vs_true_sample.csv",
 np.c_[y_true[:,0], y_pred[:,0]], delimiter=",", header="true,pred", comments="")
 mlflow.log_artifact("reports/figures/pred_vs_true_sample.csv")

 # Guarda “modelo” (solo como ejemplo; en real loggearías tu AE/DMD/KoVAE)
 mlflow.sklearn.log_model(model.encoder, artifact_path="encoder_pca")

 print(f"[OK] Run finalizado. MAE={mae:.4f} RMSE={rmse:.4f}")

if __name__ == "__main__":
 parser = argparse.ArgumentParser()
 parser.add_argument("--latent_dim", type=int, default=100)
 parser.add_argument("--epochs", type=int, default=5)
 parser.add_argument("--batch_size", type=int, default=32)
 args = parser.parse_args()
 main(**vars(args))
