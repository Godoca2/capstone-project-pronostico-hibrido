import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Limitar hilos para arranque rápido y reproducible
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Asegurar que la raíz del proyecto está en sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import numpy as np

np.random.seed(0)

print('Importando KoVAE (TF configurado)...')
from src.models.kovae import KoVAE

print('Creando KoVAE...')
k = KoVAE(spatial_dims=(157,41), latent_dim=16, beta=1.0, gamma=0.1)
print('Construyendo modelo...')
k.build()

# Datos sintéticos: 8 muestras, 3 pasos, 157x41
X_seq = np.random.rand(8, 3, 157, 41, 1).astype('float32')

print('Iniciando entrenamiento de prueba (1 epoch)...')
res = k.train_sequence(X_seq, epochs=1, batch_size=2, patience=2, learning_rate=1e-4)
print('Resultado de prueba:', res)
