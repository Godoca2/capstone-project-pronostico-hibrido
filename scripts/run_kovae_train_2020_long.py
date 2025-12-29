import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pickle
import numpy as np
from pathlib import Path
from src.utils.data_loader import load_era5_kovae
from src.models.kovae import KoVAE

OUT_DIR = Path(ROOT) / 'data' / 'models' / 'kovae_trained_step2_epoch100'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('Cargando datos ERA5 KoVAE...')
data = load_era5_kovae()
precip = data.get('precip_2020')
if precip is None:
    raise RuntimeError('clave "precip_2020" no encontrada en pickle')

print('precip shape:', precip.shape)

# Parametros de secuencia
n_steps = 7
T = precip.shape[0]
sequences = []
for i in range(T - n_steps + 1):
    seq = precip[i:i + n_steps]
    sequences.append(seq)
sequences = np.array(sequences)
print('Sequences shape:', sequences.shape)  # (N_seq, n_steps, lat, lon, 1)

# División temporal (sobre secuencias)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

n_seq = len(sequences)
n_train = int(train_ratio * n_seq)
n_val = int(val_ratio * n_seq)
X_train = sequences[:n_train]
X_val = sequences[n_train:n_train + n_val]
X_test = sequences[n_train + n_val:]
print('Split:', X_train.shape, X_val.shape, X_test.shape)

# Instanciar KoVAE
latent_dim = 64
kovae = KoVAE(spatial_dims=(precip.shape[1], precip.shape[2]), latent_dim=latent_dim, beta=1.0, gamma=0.1)
print('Construyendo KoVAE...')
kovae.build()

# Entrenamiento extenso
epochs = 100
batch_size = 16
res = kovae.train_sequence(X_train, X_val_seq=X_val, epochs=epochs, batch_size=batch_size, patience=15, learning_rate=1e-4)
print('Train result:', res)

# Guardar modelo y resumen
kovae.save(OUT_DIR)
with open(OUT_DIR / 'train_result.pkl', 'wb') as f:
    pickle.dump({'result': res}, f)

print('Entrenamiento extenso completo. Model saved to', OUT_DIR)
