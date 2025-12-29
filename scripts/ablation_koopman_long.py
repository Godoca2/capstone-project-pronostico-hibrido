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
from src.models.kovae import KoVAE


OUT_DIR = Path(ROOT) / 'reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BASE = Path(ROOT) / 'data' / 'models' / 'ablation_long'
MODEL_BASE.mkdir(parents=True, exist_ok=True)


def train_full(gamma, epochs=50, batch_size=16, latent_dim=64):
    print(f"Training full ablation: gamma={gamma}, epochs={epochs}")
    data = load_era5_kovae()
    precip = data['precip_2020']  # (T, lat, lon, 1)
    lat_dim, lon_dim = precip.shape[1], precip.shape[2]

    # sequences
    n_steps = 7
    T = precip.shape[0]
    sequences = np.array([precip[i:i+n_steps] for i in range(T - n_steps + 1)])
    n_seq = len(sequences)
    # temporal split: 70% train
    n_train = int(0.7 * n_seq)
    X_train_seq = sequences[:n_train]
    X_val_seq = sequences[n_train:n_train + int(0.15 * n_seq)]

    model = KoVAE(spatial_dims=(lat_dim, lon_dim), latent_dim=latent_dim, gamma=gamma)
    model.build()

    hist = model.train_sequence(X_train_seq, X_val_seq=X_val_seq, epochs=epochs, batch_size=batch_size)

    model_dir = MODEL_BASE / f'gamma_{gamma}'
    model_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save(model_dir / 'encoder.h5')
    model.decoder.save(model_dir / 'decoder.h5')
    model.vae.save(model_dir / 'kovae_full.h5')
    K = model.koopman_layer.get_weights()[0]
    np.save(model_dir / 'koopman_matrix.npy', K)
    with open(model_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump({'history': hist, 'gamma': gamma}, f)

    return model_dir


def main():
    # Run two trainings (gamma 0 and gamma 0.1)
    dir0 = train_full(gamma=0.0, epochs=50, batch_size=16, latent_dim=64)
    dir1 = train_full(gamma=0.1, epochs=50, batch_size=16, latent_dim=64)

    # Quick eval: compute MAE by horizon on the held-out test split
    data = load_era5_kovae()
    precip = data['precip_2020']
    n_steps = 7
    sequences = np.array([precip[i:i+n_steps] for i in range(len(precip) - n_steps + 1)])
    n_seq = len(sequences)
    n_train = int(0.7 * n_seq)
    n_val = int(0.15 * n_seq)
    X_test = sequences[n_train + n_val:]

    def load_preds(model_dir):
        from tensorflow import keras
        from src.models.kovae import KoVAE, KoopmanLayer, Sampling
        custom = {'KoopmanLayer': KoopmanLayer, 'Sampling': Sampling}
        encoder = keras.models.load_model(model_dir / 'encoder.h5', custom_objects=custom)
        decoder = keras.models.load_model(model_dir / 'decoder.h5', custom_objects=custom)
        kovae = KoVAE(spatial_dims=(precip.shape[1], precip.shape[2]), latent_dim=encoder.output[0].shape[-1])
        kovae.encoder = encoder
        kovae.decoder = decoder
        # load koopman
        K = np.load(model_dir / 'koopman_matrix.npy')
        kovae.koopman_layer = KoopmanLayer(kovae.latent_dim)
        kovae.koopman_layer.build((None, kovae.latent_dim))
        kovae.koopman_layer.set_weights([K])
        n_horizon = n_steps - 1
        N_test = X_test.shape[0]
        preds = np.zeros((N_test, n_horizon, precip.shape[1], precip.shape[2], 1))
        for i in range(N_test):
            X_init = X_test[i:i+1, 0]
            p, _ = kovae.predict_multistep(X_init, n_steps=n_horizon)
            preds[i] = p[0]
        return preds

    from src.models.kovae import KoopmanLayer
    preds0 = load_preds(dir0)
    preds1 = load_preds(dir1)
    Y_true = X_test[:, 1:, ...]
    n_horizon = n_steps - 1
    mae0 = [float(np.mean(np.abs(Y_true[:, h, ..., 0] - preds0[:, h, ..., 0]))) for h in range(n_horizon)]
    mae1 = [float(np.mean(np.abs(Y_true[:, h, ..., 0] - preds1[:, h, ..., 0]))) for h in range(n_horizon)]

    plt.figure(figsize=(8,4))
    horizons = np.arange(1, n_horizon+1)
    plt.plot(horizons, mae0, marker='o', label='gamma=0.0')
    plt.plot(horizons, mae1, marker='o', label='gamma=0.1')
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    plt.title('Ablation long: KoVAE gamma=0 vs gamma=0.1')
    plt.legend()
    plt.tight_layout()
    out = OUT_DIR / 'ablation_long_mae_by_horizon.png'
    plt.savefig(out)
    print('Saved', out)


if __name__ == '__main__':
    main()
