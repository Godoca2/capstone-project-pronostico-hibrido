"""ae_dmd.py
=================
Integración de un Autoencoder (Keras) con Dynamic Mode Decomposition (PyDMD)
para pronóstico temporal de series multivariadas de precipitación (u otras
variables). El objetivo es comprimir la dimensión espacial en un espacio
latente y modelar allí la dinámica aproximadamente lineal mediante DMD.

Motivación
----------
Los campos de precipitación tienen alta correlación espacial y redundancia.
Un Autoencoder reduce dimensión capturando patrones dominantes. La DMD,
formulación data-driven del operador de Koopman, modela evolución temporal
como combinación de modos con dinámicas exponenciales/oscillatorias.

Flujo resumido
--------------
1. Entrenar Autoencoder con datos históricos (snapshot a snapshot).
2. Codificar serie completa al espacio latente (Z).
3. Ajustar DMD sobre matriz latente (snapshots en columnas).
4. Extrapolar dinámica latente hacia el futuro (forecast).
5. Decodificar latentes pronosticadas al espacio original.

Nota sobre shapes
-----------------
Para AE denso: X shape (T, features). Para AE LSTM: (T, timesteps, features).
La codificación devuelve siempre Z shape (T, latent_dim). DMD opera sobre
Z.T (latent_dim, T) como requiere la librería PyDMD.
"""

from .ae_keras import build_autoencoder, build_autoencoder_lstm, train_autoencoder, get_encoder, encode, reconstruct
import numpy as np
from pydmd import DMD
from tensorflow import keras
import sys
import os

# Importar autoencoder de Keras
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AE_DMD:
    """Modelo híbrido Autoencoder + DMD.

    Encapsula entrenamiento del AE y posterior ajuste de DMD sobre el espacio
    latente. Permite generar pronósticos multivariados y extraer modos y
    eigenvalores (frecuencias/growth) de la dinámica latente.

    Attributes
    ----------
    latent_dim : int
        Dimensión del espacio latente.
    use_lstm : bool
        True si se emplea arquitectura LSTM en lugar de densa.
    autoencoder : keras.Model | None
        Autoencoder completo entrenado.
    encoder : keras.Model | None
        Submodelo encoder para obtener representaciones latentes.
    dmd : pydmd.DMD | None
        Instancia DMD ajustada sobre latentes.
    dmd_rank : int
        Rango SVD usado en DMD (control de complejidad).
    training_history : keras.callbacks.History | None
        Historial de entrenamiento del AE.
    """

    def __init__(
            self,
            latent_dim: int = 32,
            use_lstm: bool = False,
            dmd_rank: int = None):
        """
        Parámetros
        ----------
        latent_dim : int
            Dimensión del espacio latente
        use_lstm : bool
            Si True, usa LSTM autoencoder (requiere X con shape (samples, timesteps, features))
        dmd_rank : int
            Rango SVD para DMD. Si None, usa min(latent_dim, 20)
        """
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm
        self.dmd_rank = dmd_rank if dmd_rank is not None else min(
            latent_dim, 20)
        self.autoencoder = None
        self.encoder = None
        self.dmd = None
        self.training_history = None

    def fit_autoencoder(
            self,
            X_train: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            validation_split: float = 0.15):
        """Entrena el Autoencoder.

        Parameters
        ----------
        X_train : np.ndarray
            Datos de entrenamiento: (samples, features) o (samples, timesteps, features).
        epochs : int
            Número de épocas.
        batch_size : int
            Tamaño del batch.
        validation_split : float
            Porción reservada para validación interna de Keras.
        Returns
        -------
        AE_DMD
            Referencia al propio objeto (fluidez).
        """
        print(f"[AE-DMD] Entrenando autoencoder (LSTM={self.use_lstm}, latent_dim={self.latent_dim})...")
        self.autoencoder, self.training_history = train_autoencoder(
            X_train,
            latent_dim=self.latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            use_lstm=self.use_lstm,
            validation_split=validation_split
        )
        self.encoder = get_encoder(self.autoencoder)
        print("[AE-DMD] Autoencoder entrenado exitosamente")
        return self

    def fit_dmd(self, X_time_series: np.ndarray):
        """Ajusta DMD sobre la secuencia latente.

        Parameters
        ----------
        X_time_series : np.ndarray
            Serie temporal completa (la misma usada para forecasting futuro).

        Notes
        -----
        PyDMD espera matriz con snapshots en columnas: (latent_dim, T). Se
        codifica primero a Z (T, latent_dim) y se transpone.
        """
        if self.encoder is None:
            raise ValueError(
                "Debes entrenar el autoencoder primero con fit_autoencoder()")

        print(
            f"[AE-DMD] Codificando {X_time_series.shape[0]} snapshots al espacio latente...")
        Z = encode(self.autoencoder, X_time_series)

        # DMD espera snapshots en columnas: (latent_dim, T)
        snapshots = Z.T
        print(
            f"[AE-DMD] Aplicando DMD sobre snapshots latentes: shape={snapshots.shape}")

        self.dmd = DMD(svd_rank=self.dmd_rank)
        self.dmd.fit(snapshots)

        print(
            f"[AE-DMD] DMD ajustado: {len(self.dmd.modes.T)} modos, rango={self.dmd_rank}")
        return self

    def fit(self, X_train: np.ndarray, X_time_series: np.ndarray = None,
            epochs: int = 50, batch_size: int = 32):
        """Ejecuta pipeline completo (AE + DMD).

        Parameters
        ----------
        X_train : np.ndarray
            Datos para entrenar el AE.
        X_time_series : np.ndarray | None
            Serie para DMD; si None se reutiliza X_train.
        epochs : int
            Épocas de entrenamiento del AE.
        batch_size : int
            Tamaño de batch.
        Returns
        -------
        AE_DMD
            Instancia entrenada.
        """
        # 1. Entrenar autoencoder
        self.fit_autoencoder(X_train, epochs=epochs, batch_size=batch_size)

        # 2. Ajustar DMD
        if X_time_series is None:
            X_time_series = X_train
        self.fit_dmd(X_time_series)

        return self

    def forecast(self, steps: int = 10, method: str = 'dmd') -> np.ndarray:
        """Genera pronóstico multivariado.

        Parameters
        ----------
        steps : int
            Pasos futuros.
        method : str
            'dmd' para dinámica latente; 'last' baseline repitiendo último estado.
        Returns
        -------
        np.ndarray
            Pronóstico en espacio original.
        """
        if self.dmd is None or self.autoencoder is None:
            raise ValueError("Debes ajustar el modelo primero con fit()")

        if method == 'dmd':
            # Extender DMD hacia el futuro
            print(f"[AE-DMD] Forecasting {steps} pasos con DMD...")
            time_dynamics = self.dmd.dynamics

            # Última ventana temporal conocida
            last_time_idx = self.dmd.original_time['t'][-1]
            future_times = np.arange(
                last_time_idx + 1, last_time_idx + steps + 1)

            # Reconstruir con DMD en tiempos futuros
            b = self.dmd.amplitudes
            omega = self.dmd.eigs
            modes = self.dmd.modes

            Z_future = []
            for t in future_times:
                # DMD reconstruction: Φ * diag(ω^t) * b
                z_t = (modes @ np.diag(np.power(omega, t)) @ b).real
                Z_future.append(z_t)

            Z_future = np.array(Z_future)  # shape (steps, latent_dim)

        elif method == 'last':
            # Baseline: repetir último estado latente
            Z_last = self.dmd.reconstructed_data.real.T[-1:]
            Z_future = np.repeat(Z_last, steps, axis=0)
        else:
            raise ValueError(f"Método desconocido: {method}")

        # Decodificar al espacio original
        print(
            f"[AE-DMD] Decodificando {Z_future.shape[0]} estados latentes...")
        X_forecast = reconstruct(self.autoencoder, Z_future)

        return X_forecast

    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """Calcula MSE de reconstrucción del AE sobre X."""
        X_reconstructed = reconstruct(self.autoencoder, X)
        mse = np.mean((X - X_reconstructed) ** 2)
        return mse

    def get_dmd_modes(self) -> np.ndarray:
        """Retorna modos espaciales (columnas de Φ) de la DMD latente."""
        if self.dmd is None:
            raise ValueError("DMD no ajustado")
        return self.dmd.modes

    def get_dmd_eigenvalues(self) -> np.ndarray:
        """Eigenvalores de DMD (frecuencias y tasas de crecimiento/decay)."""
        if self.dmd is None:
            raise ValueError("DMD no ajustado")
        return self.dmd.eigs
