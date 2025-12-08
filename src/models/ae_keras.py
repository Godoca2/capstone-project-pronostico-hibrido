"""ae_keras.py
=================
Autoencoders densos y LSTM en Keras para reducir dimensionalidad de series
climáticas multivariadas (p.ej. matrices de precipitación). Se emplean como
parte de un pipeline híbrido AE + DMD.

Resumen de funciones
--------------------
build_autoencoder_lstm : Construye AE LSTM para ventanas (timesteps, features).
build_autoencoder : AE denso para snapshots (features).
train_autoencoder : Entrena AE elegido (denso o LSTM) con early stopping.
get_encoder : Extrae submodelo encoder.
encode : Codifica datos (obtiene latentes).
reconstruct : Reconstruye datos desde latentes con decoder.

Decisiones de diseño
--------------------
* Activación ReLU en capas ocultas; salida lineal para regresión continua.
* EarlyStopping con paciencia=10 para evitar sobreajuste.
* Latent layer nombrada 'latent_space' para extracción consistente.

Limitaciones / futuras mejoras
------------------------------
* No incluye regularización explícita (Dropout / L2) → podría añadirse.
* Para LSTM solo se soporta arquitectura fija (128→64). Podría hacerse
 parametrizable.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


def build_autoencoder_lstm(
        timesteps: int,
        n_features: int,
        latent_dim: int = 64) -> models.Model:
    """Construye un Autoencoder LSTM.

    Parameters
    ----------
    timesteps : int
    Longitud de la ventana temporal.
    n_features : int
    Número de variables (estaciones) por timestep.
    latent_dim : int
    Dimensión del espacio latente.
    Returns
    -------
    keras.Model
    Modelo autoencoder LSTM compilado.
    """
    # Encoder
    input_layer = layers.Input(shape=(timesteps, n_features))
    encoded = layers.LSTM(
        128,
        activation='relu',
        return_sequences=True)(input_layer)
    encoded = layers.LSTM(
        64,
        activation='relu',
        return_sequences=False)(encoded)
    latent = layers.Dense(
        latent_dim,
        activation='relu',
        name='latent_space')(encoded)

    # Decoder
    decoded = layers.RepeatVector(timesteps)(latent)
    decoded = layers.LSTM(
        64,
        activation='relu',
        return_sequences=True)(decoded)
    decoded = layers.LSTM(
        128,
        activation='relu',
        return_sequences=True)(decoded)
    output = layers.TimeDistributed(layers.Dense(n_features))(decoded)

    autoencoder = models.Model(input_layer, output)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return autoencoder


def build_autoencoder(input_dim: int, latent_dim: int = 32) -> models.Model:
    """Construye un Autoencoder denso.

    Parameters
    ----------
    input_dim : int
    Número de características en cada snapshot.
    latent_dim : int
    Tamaño del embedding latente.
    Returns
    -------
    keras.Model
    Modelo autoencoder denso compilado.
    """
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation="relu")(input_layer)
    encoded = layers.Dense(128, activation="relu")(encoded)
    encoded = layers.Dense(
        latent_dim,
        activation="relu",
        name='latent_space')(encoded)

    decoded = layers.Dense(128, activation="relu")(encoded)
    decoded = layers.Dense(256, activation="relu")(decoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse", metrics=['mae'])
    return autoencoder


def train_autoencoder(
        X_train: np.ndarray,
        latent_dim: int = 32,
        epochs: int = 50,
        batch_size: int = 32,
        use_lstm: bool = False,
        validation_split: float = 0.15):
    """Entrena el autoencoder especificado.

    Parameters
    ----------
    X_train : np.ndarray
        Datos: (samples, features) o (samples, timesteps, features).
    latent_dim : int
        Dimensión del espacio latente.
    epochs : int
        Número de épocas.
    batch_size : int
        Tamaño de batch.
    use_lstm : bool
        Activa arquitectura LSTM.
    validation_split : float
        Porción de datos para validación.
    Returns
    -------
    tuple[keras.Model, keras.callbacks.History]
        Modelo entrenado y objeto History.
    """
    if use_lstm:
        if len(X_train.shape) != 3:
            raise ValueError(
                "Para LSTM, X_train debe tener shape (samples, timesteps, features)")
        timesteps, n_features = X_train.shape[1], X_train.shape[2]
        model = build_autoencoder_lstm(timesteps, n_features, latent_dim)
        print(f"Entrenando LSTM Autoencoder: timesteps={timesteps}, features={n_features}, latent_dim={latent_dim}")
    else:
        if len(X_train.shape) != 2:
            raise ValueError(
                "Para AE denso, X_train debe tener shape (samples, features)")
        input_dim = X_train.shape[1]
        model = build_autoencoder(input_dim, latent_dim)
        print(f"Entrenando Dense Autoencoder: input_dim={input_dim}, latent_dim={latent_dim}")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)

    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history


def get_encoder(model: models.Model) -> models.Model:
    """Devuelve submodelo encoder usando capa 'latent_space'."""
    latent_layer = model.get_layer('latent_space')
    encoder = models.Model(model.input, latent_layer.output)
    return encoder


def encode(model: models.Model, X: np.ndarray,
           batch_size: int = 32) -> np.ndarray:
    """Retorna representaciones latentes del autoencoder."""
    encoder = get_encoder(model)
    return encoder.predict(X, batch_size=batch_size, verbose=0)


def reconstruct(model: models.Model, X: np.ndarray,
                batch_size: int = 32) -> np.ndarray:
    """Reconstruye datos pasando por el autoencoder completo."""
    return model.predict(X, batch_size=batch_size, verbose=0)


if __name__ == "__main__":
    # Ejemplo rápido
    X = np.random.rand(500, 50)
    model, hist = train_autoencoder(X, latent_dim=10, epochs=5)
    model.save("data/models/ae_minimal.h5")
    print("[OK] Entrenamiento finalizado y modelo guardado.")
