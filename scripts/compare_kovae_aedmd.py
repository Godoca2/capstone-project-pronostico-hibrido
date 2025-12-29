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
from sklearn.metrics import mean_absolute_error
from src.utils.data_loader import load_era5_kovae, load_forecast_results
from src.models.kovae import KoVAE, Sampling, KoopmanLayer
from tensorflow import keras


MODEL_DIR = Path(ROOT) / 'data' / 'models' / 'kovae_trained_step2_epoch100'
OUT_DIR = Path(ROOT) / 'reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
R2_IN = Path(ROOT) / 'data' / 'processed' / 'kovae_r2_map_h1.pkl'
WORST_CELLS_CSV = Path(ROOT) / 'reports' / 'figures' / 'kovae_worst_cells_examples.csv'


def load_aedmd_forecasts():
    results = load_forecast_results()
    if 'forecast_results' not in results:
        raise RuntimeError('forecast_results key missing in forecast file')
    fr = results['forecast_results']
    # Expect fr to be dict horizon->array (N_test, lat, lon[,1])
    return fr, results.get('y_test_real', None)


def main():
    print('Cargando datos...')
    data = load_era5_kovae()
    precip = data['precip_2020']  # (T, lat, lon, 1)
    lat_dim, lon_dim = precip.shape[1], precip.shape[2]

    n_steps = 7
    T = precip.shape[0]
    sequences = np.array([precip[i:i+n_steps] for i in range(T - n_steps + 1)])
    train_ratio = 0.7
    val_ratio = 0.15
    n_seq = len(sequences)
    n_train = int(train_ratio * n_seq)
    n_val = int(val_ratio * n_seq)
    X_test = sequences[n_train + n_val:]
    Y_true = X_test[:, 1:, ...]
    N_test = X_test.shape[0]
    n_horizon = n_steps - 1

    # KoVAE preds
    print('Generando predicciones KoVAE...')
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

    all_preds_kovae = np.zeros((N_test, n_horizon, lat_dim, lon_dim), dtype=float)
    for i in range(N_test):
        X_init = X_test[i:i+1, 0]
        preds, _ = kovae.predict_multistep(X_init, n_steps=n_horizon)
        all_preds_kovae[i] = preds[0, ..., 0]

    # AE+DMD forecasts
    print('Cargando AE+DMD forecasts...')
    fr_dict, y_test_real = load_aedmd_forecasts()
    # Construct array with shape (N_test, n_horizon, lat, lon)
    all_preds_aedmd = np.zeros_like(all_preds_kovae)
    for h in range(n_horizon):
        # try several key formats: 'h1', '1', 1
        candidates = [f'h{h+1}', str(h+1), (h+1)]
        arr = None
        for k in candidates:
            if k in fr_dict:
                arr = fr_dict[k]
                break
        if arr is None:
            # fallback: if AE+DMD doesn't have this horizon, fill with np.nan
            arr = np.full((N_test, lat_dim, lon_dim), np.nan)
        # if stored as dict with metadata, extract first array-like value
        if isinstance(arr, dict):
            for v in arr.values():
                if hasattr(v, 'ndim'):
                    arr = v
                    break
        # ensure shape
        if hasattr(arr, 'ndim') and arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        all_preds_aedmd[:, h] = arr
        # ensure shape
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        all_preds_aedmd[:, h] = arr

    # Compute MAE maps h=1
    h0 = 0
    mae_kovae = np.mean(np.abs(Y_true[:, h0, ..., 0] - all_preds_kovae[:, h0, ...]), axis=0)
    mae_aedmd = np.mean(np.abs(Y_true[:, h0, ..., 0] - all_preds_aedmd[:, h0, ...]), axis=0)
    diff = mae_kovae - mae_aedmd

    plt.figure(figsize=(8,4))
    sns.heatmap(mae_kovae, cmap='viridis')
    plt.title('MAE KoVAE h=1')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mae_map_kovae_h1.png')

    plt.figure(figsize=(8,4))
    sns.heatmap(mae_aedmd, cmap='viridis')
    plt.title('MAE AE+DMD h=1')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mae_map_aedmd_h1.png')

    plt.figure(figsize=(8,4))
    sns.heatmap(diff, cmap='bwr', center=0)
    plt.title('MAE Difference (KoVAE - AE+DMD) h=1')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mae_diff_kovae_minus_aedmd_h1.png')
    print('Saved MAE maps and difference')

    # Save numeric arrays
    np.save(Path(ROOT) / 'data' / 'processed' / 'mae_kovae_h1.npy', mae_kovae)
    np.save(Path(ROOT) / 'data' / 'processed' / 'mae_aedmd_h1.npy', mae_aedmd)
    np.save(Path(ROOT) / 'data' / 'processed' / 'mae_diff_h1.npy', diff)

    # Bias maps
    bias_kovae = np.mean(all_preds_kovae[:, h0, ...] - Y_true[:, h0, ..., 0], axis=0)
    bias_aedmd = np.mean(all_preds_aedmd[:, h0, ...] - Y_true[:, h0, ..., 0], axis=0)
    bias_diff = bias_kovae - bias_aedmd

    plt.figure(figsize=(8,4))
    sns.heatmap(bias_kovae, cmap='RdBu', center=0)
    plt.title('Bias KoVAE h=1')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_map_kovae_h1.png')

    plt.figure(figsize=(8,4))
    sns.heatmap(bias_aedmd, cmap='RdBu', center=0)
    plt.title('Bias AE+DMD h=1')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_map_aedmd_h1.png')

    plt.figure(figsize=(8,4))
    sns.heatmap(bias_diff, cmap='bwr', center=0)
    plt.title('Bias Difference (KoVAE - AE+DMD) h=1')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_diff_kovae_minus_aedmd_h1.png')
    print('Saved bias maps and difference')

    np.save(Path(ROOT) / 'data' / 'processed' / 'bias_kovae_h1.npy', bias_kovae)
    np.save(Path(ROOT) / 'data' / 'processed' / 'bias_aedmd_h1.npy', bias_aedmd)
    np.save(Path(ROOT) / 'data' / 'processed' / 'bias_diff_h1.npy', bias_diff)

    # Load r2 map and find worst cells
    if R2_IN.exists():
        with open(R2_IN, 'rb') as f:
            r2_map = pickle.load(f)
        flat_idx = np.argsort(r2_map.flatten())[:10]
        coords = [(int(idx // lon_dim), int(idx % lon_dim)) for idx in flat_idx]
    else:
        coords = []

    # For each worst cell, extract top error samples and save rows
    rows = []
    for (i, j) in coords:
        errors = np.abs(all_preds_kovae[:, h0, i, j] - Y_true[:, h0, i, j, 0])
        top_idx = np.argsort(errors)[-5:][::-1]
        for ti in top_idx:
            rows.append({'lat_idx': i, 'lon_idx': j, 'test_sample': int(ti), 'abs_error': float(errors[ti]),
                         'pred_kovae': float(all_preds_kovae[ti, h0, i, j]), 'truth': float(Y_true[ti, h0, i, j, 0])})

    # save CSV
    import csv
    with open(WORST_CELLS_CSV, 'w', newline='') as csvfile:
        fieldnames = ['lat_idx', 'lon_idx', 'test_sample', 'abs_error', 'pred_kovae', 'truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Saved worst cells examples to', WORST_CELLS_CSV)


if __name__ == '__main__':
    main()
