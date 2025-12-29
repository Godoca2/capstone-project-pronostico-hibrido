import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from src.utils.data_loader import load_era5_kovae
from src.models.kovae import KoVAE, KoopmanLayer, Sampling


OUT_DIR = Path(ROOT) / 'reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BASE = Path(ROOT) / 'data' / 'models' / 'ablation'
MODEL_BASE.mkdir(parents=True, exist_ok=True)


def run_ablation(subset_size=120, epochs=6, batch_size=8, gamma=0.0, name='gamma0'):
    print(f"Running ablation: {name}, gamma={gamma}")
    data = load_era5_kovae()
    precip = data['precip_2020']  # (T, lat, lon, 1)
    lat_dim, lon_dim = precip.shape[1], precip.shape[2]

    # Build sequences
    n_steps = 7
    T = precip.shape[0]
    sequences = np.array([precip[i:i+n_steps] for i in range(T - n_steps + 1)])

    # use small subset for quick run
    X = sequences[:subset_size]
    # split small val
    n_val = int(0.15 * len(X))
    X_train = X[:-n_val]
    X_val = X[-n_val:]

    # create model
    model = KoVAE(spatial_dims=(lat_dim, lon_dim), latent_dim=32, gamma=gamma)
    model.build()

    # train sequence quickly
    hist = model.train_sequence(X_train, X_val_seq=X_val, epochs=epochs, batch_size=batch_size)

    # save components
    model_dir = MODEL_BASE / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save(model_dir / 'encoder.h5')
    model.decoder.save(model_dir / 'decoder.h5')
    model.vae.save(model_dir / 'kovae_full.h5')
    # koopman matrix
    K = model.koopman_layer.get_weights()[0]
    np.save(model_dir / 'koopman_matrix.npy', K)

    # quick eval on a held-out larger test set
    sequences_full = sequences
    test = sequences_full[subset_size:subset_size+60]
    N_test = test.shape[0]
    n_horizon = n_steps - 1
    preds = np.zeros((N_test, n_horizon, lat_dim, lon_dim, 1))
    for i in range(N_test):
        X_init = test[i:i+1, 0]
        p, _ = model.predict_multistep(X_init, n_steps=n_horizon)
        preds[i] = p[0]

    Y_true = test[:, 1:, ...]

    # compute MAE per horizon
    maes = []
    for h in range(n_horizon):
        mae = np.mean(np.abs(Y_true[:, h, ..., 0] - preds[:, h, ..., 0]))
        maes.append(mae)

    # save results
    with open(model_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump({'mae_by_horizon': maes, 'history': hist}, f)

    return {'name': name, 'mae': maes, 'model_dir': model_dir}


def main():
    # run two experiments quickly
    r1 = run_ablation(subset_size=120, epochs=6, batch_size=8, gamma=0.0, name='gamma0')
    r2 = run_ablation(subset_size=120, epochs=6, batch_size=8, gamma=0.1, name='gamma0.1')

    # plot comparison
    horizons = np.arange(1, 7)
    plt.figure(figsize=(8,4))
    plt.plot(horizons, r1['mae'], marker='o', label=r1['name'])
    plt.plot(horizons, r2['mae'], marker='o', label=r2['name'])
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    plt.title('Ablation: KoVAE gamma=0 vs gamma=0.1 (quick run)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'ablation_koopman_mae_by_horizon.png')
    print('Saved ablation figure to', OUT_DIR / 'ablation_koopman_mae_by_horizon.png')


if __name__ == '__main__':
    main()
