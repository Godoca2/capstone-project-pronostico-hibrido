import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from src.utils.data_loader import load_era5_kovae
from src.models.kovae import KoVAE, Sampling, KoopmanLayer
from tensorflow import keras


# Paths
MODEL_DIR = Path(ROOT) / 'data' / 'models' / 'kovae_trained_step2_epoch100'
OUT_DIR = Path(ROOT) / 'reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
R2_OUT = Path(ROOT) / 'data' / 'processed' / 'kovae_r2_map_h1.pkl'
BIAS_OUT = Path(ROOT) / 'data' / 'processed' / 'kovae_bias_macrozone.pkl'


def safe_r2(y_true, y_pred):
    # If constant truth (zero variance), return np.nan
    if np.allclose(y_true, y_true.flat[0]):
        return np.nan
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def main():
    print('Cargando datos y modelo...')
    data = load_era5_kovae()
    precip = data['precip_2020']  # (T, lat, lon, 1)
    lat_dim, lon_dim = precip.shape[1], precip.shape[2]

    # sequences
    n_steps = 7
    T = precip.shape[0]
    sequences = np.array([precip[i:i+n_steps] for i in range(T - n_steps + 1)])
    train_ratio = 0.7
    val_ratio = 0.15
    n_seq = len(sequences)
    n_train = int(train_ratio * n_seq)
    n_val = int(val_ratio * n_seq)
    X_test = sequences[n_train + n_val:]
    print('X_test shape:', X_test.shape)

    # load model
    custom = {'Sampling': Sampling, 'KoopmanLayer': KoopmanLayer}
    vae = keras.models.load_model(MODEL_DIR / 'kovae_full.h5', custom_objects=custom)
    encoder = keras.models.load_model(MODEL_DIR / 'encoder.h5', custom_objects=custom)
    decoder = keras.models.load_model(MODEL_DIR / 'decoder.h5', custom_objects=custom)
    kovae = KoVAE(spatial_dims=(lat_dim, lon_dim), latent_dim=encoder.output[0].shape[-1])
    kovae.vae = vae
    kovae.encoder = encoder
    kovae.decoder = decoder
    kovae.koopman_layer = KoopmanLayer(kovae.latent_dim)
    kovae.koopman_layer.build((None, kovae.latent_dim))
    K_matrix = np.load(MODEL_DIR / 'koopman_matrix.npy')
    kovae.koopman_layer.set_weights([K_matrix])

    n_horizon = n_steps - 1
    N_test = X_test.shape[0]

    print('Generando predicciones KoVAE...')
    all_preds = np.zeros((N_test, n_horizon, lat_dim, lon_dim, 1), dtype=float)
    for i in range(N_test):
        X_init = X_test[i:i+1, 0]
        preds, _ = kovae.predict_multistep(X_init, n_steps=n_horizon)
        all_preds[i] = preds[0]

    Y_true = X_test[:, 1:, ...]

    # A) R2 por punto espacial, horizon 1
    h0 = 0
    r2_map = np.zeros((lat_dim, lon_dim), dtype=float)
    for i in range(lat_dim):
        for j in range(lon_dim):
            y = Y_true[:, h0, i, j, 0]
            yp = all_preds[:, h0, i, j, 0]
            r2_map[i, j] = safe_r2(y, yp)

    # save and plot
    with open(R2_OUT, 'wb') as f:
        pickle.dump(r2_map, f)
    plt.figure(figsize=(8,4))
    sns.heatmap(r2_map, cmap='RdBu', center=0, vmin=-1, vmax=1)
    plt.title('R2 map (KoVAE) - Horizon 1')
    plt.xlabel('Lon index')
    plt.ylabel('Lat index')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'r2_map_h1_kovae.png')
    print('Saved R2 map to', OUT_DIR / 'r2_map_h1_kovae.png')

    # B) Bias por macrozona
    nlat = lat_dim
    band = nlat // 3
    zones = {'Norte': slice(0, band), 'Centro': slice(band, 2*band), 'Sur': slice(2*band, nlat)}
    bias_by_zone = {}
    for name, sl in zones.items():
        bias_by_zone[name] = {}
        for h in range(n_horizon):
            err = all_preds[:, h, sl, :, 0] - Y_true[:, h, sl, :, 0]
            bias = float(np.mean(err))
            bias_by_zone[name][f'h{h+1}'] = bias

    with open(BIAS_OUT, 'wb') as f:
        pickle.dump(bias_by_zone, f)

    # plot bias per horizon
    plt.figure(figsize=(8,4))
    horizons = list(range(1, n_horizon+1))
    for name in zones.keys():
        vals = [bias_by_zone[name][f'h{h}'] for h in horizons]
        plt.plot(horizons, vals, marker='o', label=name)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Horizon')
    plt.ylabel('Bias (prediction - truth)')
    plt.title('Bias por macrozona y horizonte')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_macrozone_by_horizon.png')
    print('Saved bias by horizon plot')

    # timeseries bias for h=1
    plt.figure(figsize=(10,6))
    for name, sl in zones.items():
        err_ts = (all_preds[:, 0, sl, :, 0] - Y_true[:, 0, sl, :, 0]).mean(axis=(1,2))
        plt.plot(err_ts, label=name)
    plt.xlabel('Test sample index')
    plt.ylabel('Bias (mean over region)')
    plt.title('Bias time series by macrozona (h=1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_macrozone_timeseries_h1.png')
    print('Saved bias timeseries plot')

    print('\nDiagnosis outputs saved in:', OUT_DIR)


if __name__ == '__main__':
    main()
