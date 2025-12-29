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
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.data_loader import load_era5_kovae
from src.models.kovae import KoVAE, Sampling, KoopmanLayer
from tensorflow import keras

# Paths
MODEL_DIR = Path(ROOT) / 'data' / 'models' / 'kovae_trained_step2_epoch100'
OUT_DIR = Path(ROOT) / 'reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUT = Path(ROOT) / 'data' / 'processed' / 'kovae_evaluation_metrics.pkl'

# Load prepared ERA5
print('Cargando ERA5 KoVAE...')
data = load_era5_kovae()
precip = data['precip_2020']  # (T, lat, lon, 1)
lat_dim, lon_dim = precip.shape[1], precip.shape[2]

# Make sequences
n_steps = 7
T = precip.shape[0]
sequences = np.array([precip[i:i+n_steps] for i in range(T - n_steps + 1)])
# split
train_ratio = 0.7
val_ratio = 0.15
n_seq = len(sequences)
n_train = int(train_ratio * n_seq)
n_val = int(val_ratio * n_seq)
X_test = sequences[n_train + n_val:]
print('X_test shape:', X_test.shape)

# Load KoVAE model components
print('Cargando KoVAE...')
custom = {'Sampling': Sampling, 'KoopmanLayer': KoopmanLayer}
vae = keras.models.load_model(MODEL_DIR / 'kovae_full.h5', custom_objects=custom)
encoder = keras.models.load_model(MODEL_DIR / 'encoder.h5', custom_objects=custom)
decoder = keras.models.load_model(MODEL_DIR / 'decoder.h5', custom_objects=custom)

# rebuild instance
kovae = KoVAE(spatial_dims=(lat_dim, lon_dim), latent_dim=encoder.output[0].shape[-1])
kovae.vae = vae
kovae.encoder = encoder
kovae.decoder = decoder
kovae.koopman_layer = KoopmanLayer(kovae.latent_dim)
kovae.koopman_layer.build((None, kovae.latent_dim))
K_matrix = np.load(MODEL_DIR / 'koopman_matrix.npy')
kovae.koopman_layer.set_weights([K_matrix])

# Predict for all X_test
n_horizon = n_steps - 1
N_test = X_test.shape[0]
print('Generando predicciones KoVAE...')
all_preds = np.zeros((N_test, n_horizon, lat_dim, lon_dim, 1), dtype=float)
for i in range(N_test):
    X_init = X_test[i:i+1, 0]
    preds, _ = kovae.predict_multistep(X_init, n_steps=n_horizon)
    all_preds[i] = preds[0]

# Ground truth
Y_true = X_test[:, 1:, ...]

# Compute per-horizon metrics and print
print('\nMetrics por horizonte (KoVAE):')
metrics = {}
for h in range(n_horizon):
    y = Y_true[:, h].reshape(N_test, -1)
    y_pred = all_preds[:, h].reshape(N_test, -1)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    metrics[f'h{h+1}'] = {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}
    print(f"h{h+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

# Save metrics
with open(METRICS_OUT, 'wb') as f:
    pickle.dump(metrics, f)
print('\nMetrics saved to', METRICS_OUT)

# 1) Spatial error map for h=1 (mean absolute error across test samples)
h0 = 0
abs_err_maps = np.mean(np.abs(Y_true[:, h0, ..., 0] - all_preds[:, h0, ..., 0]), axis=0)
plt.figure(figsize=(8,4))
sns.heatmap(abs_err_maps, cmap='viridis')
plt.title('Mean Absolute Error Map (KoVAE) - Horizon 1')
plt.xlabel('Lon index')
plt.ylabel('Lat index')
plt.tight_layout()
plt.savefig(OUT_DIR / 'spatial_mae_map_h1_kovae.png')
print('Saved spatial MAE map')

# 2) Scatter plot GT vs Pred for h=1 (sampled points)
y_gt = Y_true[:, h0].reshape(-1)
y_pr = all_preds[:, h0].reshape(-1)
# sample for plotting if too many
idx = np.random.choice(len(y_gt), size=min(20000, len(y_gt)), replace=False)
plt.figure(figsize=(6,6))
plt.hexbin(y_gt[idx], y_pr[idx], gridsize=80, cmap='inferno', mincnt=1)
plt.plot([y_gt.min(), y_gt.max()], [y_gt.min(), y_gt.max()], 'k--')
plt.xlabel('Ground Truth (mm/d)')
plt.ylabel('KoVAE Prediction (mm/d)')
plt.title('Scatter GT vs KoVAE (h=1)')
plt.colorbar(label='counts')
plt.tight_layout()
plt.savefig(OUT_DIR / 'scatter_gt_vs_kovae_h1.png')
print('Saved scatter plot')

# 3) Series by macrozona: simple lat split into 3 bands
nlat = lat_dim
band = nlat // 3
zones = {'Norte': slice(0, band), 'Centro': slice(band, 2*band), 'Sur': slice(2*band, nlat)}
plt.figure(figsize=(10,6))
for name, sl in zones.items():
    # compute mean over spatial dims for each test sequence and horizon 1
    gt_mean = Y_true[:, 0, sl, :, 0].mean(axis=(1,2))
    pr_mean = all_preds[:, 0, sl, :, 0].mean(axis=(1,2))
    plt.plot(gt_mean, label=f'GT {name}')
    plt.plot(pr_mean, '--', label=f'Pr {name}')
plt.legend()
plt.title('Series por macrozona (mean over region) - Horizon 1 across test samples')
plt.xlabel('Test sample index')
plt.ylabel('Precip (mm/d)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'series_macrozone_h1.png')
print('Saved macrozone series')

print('\nAll figures saved in', OUT_DIR)
print('Done')
