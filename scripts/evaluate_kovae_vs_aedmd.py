import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.data_loader import load_era5_kovae
from src.models.kovae import KoVAE

# Paths
MODEL_DIR = Path(ROOT) / 'data' / 'models' / 'kovae_trained_step2_epoch100'
OUT_DIR = Path(ROOT) / 'reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUT = Path(ROOT) / 'data' / 'processed' / 'kovae_evaluation_metrics.pkl'

# Load data
print('Cargando ERA5 KoVAE...')
data = load_era5_kovae()
precip = data['precip_2020']  # (T, lat, lon, 1)

# Build sequences
n_steps = 7
T = precip.shape[0]
sequences = []
for i in range(T - n_steps + 1):
    seq = precip[i:i + n_steps]
    sequences.append(seq)
sequences = np.array(sequences)

# Split as before
train_ratio = 0.7
val_ratio = 0.15
n_seq = len(sequences)
n_train = int(train_ratio * n_seq)
n_val = int(val_ratio * n_seq)
X_train = sequences[:n_train]
X_val = sequences[n_train:n_train + n_val]
X_test = sequences[n_train + n_val:]
print('X_test shape:', X_test.shape)

# Load AE+DMD forecasts
FORECAST_PATH = Path(ROOT) / 'data' / 'processed' / 'forecast_results_2020.pkl'
with open(FORECAST_PATH, 'rb') as f:
    aedmd = pickle.load(f)

# Expecting aedmd['forecast_results'][h] arrays or similar. Inspect keys
print('AEDMD keys:', list(aedmd.keys()))

# Load KoVAE model
print('Cargando KoVAE...')
from tensorflow import keras
from src.models.kovae import Sampling, KoopmanLayer

# Cargar modelos con objetos personalizados
vae_path = MODEL_DIR / 'kovae_full.h5'
enc_path = MODEL_DIR / 'encoder.h5'
dec_path = MODEL_DIR / 'decoder.h5'

custom = {'Sampling': Sampling, 'KoopmanLayer': KoopmanLayer}
vae = keras.models.load_model(vae_path, custom_objects=custom)
encoder = keras.models.load_model(enc_path, custom_objects=custom)
decoder = keras.models.load_model(dec_path, custom_objects=custom)

# Reconstruir instancia KoVAE y asignar componentes
kovae = KoVAE(spatial_dims=(precip.shape[1], precip.shape[2]), latent_dim=encoder.output[0].shape[-1])
kovae.vae = vae
kovae.encoder = encoder
kovae.decoder = decoder
kovae.koopman_layer = KoopmanLayer(kovae.latent_dim)
kovae.koopman_layer.build((None, kovae.latent_dim))
K_matrix = np.load(MODEL_DIR / 'koopman_matrix.npy')
kovae.koopman_layer.set_weights([K_matrix])

# Predict with KoVAE for each initial state in X_test: use predict_multistep with n_steps-1 horizons
n_horizon = n_steps - 1
all_preds = []  # shape (N_test, h, lat, lon, 1)
for i in range(X_test.shape[0]):
    X_init = X_test[i:i+1, 0]  # first timestep as initial state, shape (1, lat, lon, 1)
    preds, uncert = kovae.predict_multistep(X_init, n_steps=n_horizon)
    all_preds.append(preds[0])
all_preds = np.array(all_preds)
print('KoVAE preds shape:', all_preds.shape)

# Ground truth horizons: for each sequence, ground truth at t+1..t+h
Y_true = X_test[:, 1:, ...]  # shape (N_test, h, lat, lon, 1)

# Compute metrics per horizon (flatten spatial dims)
N_test, H, lat, lon, ch = Y_true.shape
metrics = {'kovae': {}, 'aedmd': {}}

# Prepare AEDMD predictions: try to match shape
# If aedmd contains 'forecast_results' with keys like 'h1','h2', map them
aedmd_preds = None
if 'forecast_results' in aedmd:
    fr = aedmd['forecast_results']
    # Attempt to build array (N_test, H, lat, lon, 1)
    # If fr[h] is shaped (N_test, lat, lon), stack
    preds_list = []
    for h in range(1, H+1):
        key = f'h{h}'
        if key in fr:
            arr = fr[key]
            if arr.ndim == 3:
                arr = arr[..., np.newaxis]
            preds_list.append(arr)
    if len(preds_list) == H:
        aedmd_preds = np.stack(preds_list, axis=1)
        print('AEDMD preds shape built:', aedmd_preds.shape)

# If aedmd_preds not available, skip AEDMD comparison
for h in range(H):
    y = Y_true[:, h].reshape(N_test, -1)
    y_pred_k = all_preds[:, h].reshape(N_test, -1)
    mae_k = mean_absolute_error(y, y_pred_k)
    rmse_k = np.sqrt(mean_squared_error(y, y_pred_k))
    r2_k = r2_score(y, y_pred_k)
    metrics['kovae'][f'h{h+1}'] = {'mae': float(mae_k), 'rmse': float(rmse_k), 'r2': float(r2_k)}

    if aedmd_preds is not None:
        y_pred_a = aedmd_preds[:, h].reshape(N_test, -1)
        mae_a = mean_absolute_error(y, y_pred_a)
        rmse_a = np.sqrt(mean_squared_error(y, y_pred_a))
        r2_a = r2_score(y, y_pred_a)
        metrics['aedmd'][f'h{h+1}'] = {'mae': float(mae_a), 'rmse': float(rmse_a), 'r2': float(r2_a)}

# Save metrics
with open(METRICS_OUT, 'wb') as f:
    pickle.dump(metrics, f)
print('Metrics saved to', METRICS_OUT)

# Plot MAE per horizon
horizons = list(range(1, H+1))
mae_k = [metrics['kovae'][f'h{h}']['mae'] for h in horizons]
plt.figure()
plt.plot(horizons, mae_k, marker='o', label='KoVAE')
if 'aedmd' in metrics and metrics['aedmd']:
    mae_a = [metrics['aedmd'][f'h{h}']['mae'] for h in horizons]
    plt.plot(horizons, mae_a, marker='x', label='AE+DMD')
plt.xlabel('Horizonte (días)')
plt.ylabel('MAE (mm/día)')
plt.title('MAE por horizonte: KoVAE vs AE+DMD')
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / 'mae_by_horizon_kovae_vs_aedmd.png')
print('Figure saved')

print('Done')
