"""Shared-encoder normal-behavior model."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(sequence_length: int, input_dim: int, target_dim: int, latent_dim: int = 32) -> keras.Model:
    """Build an LSTM encoder with room-specific heads and one reconstruction head."""
    inputs = keras.Input(shape=(sequence_length, input_dim), name="sensor_window")
    encoded = layers.LSTM(64, return_sequences=True, name="temporal_encoder_1")(inputs)
    encoded = layers.LayerNormalization(name="encoder_norm")(encoded)
    latent = layers.LSTM(latent_dim, name="shared_latent")(encoded)

    room_outputs = []
    for room_index in range(target_dim):
        head = layers.Dense(16, activation="relu", name=f"room_{room_index}_hidden")(latent)
        room_outputs.append(layers.Dense(1, name=f"room_{room_index}_prediction")(head))
    room_predictions = layers.Concatenate(name="room_predictions")(room_outputs)

    reconstruction_hidden = layers.Dense(64, activation="relu", name="reconstruction_hidden")(latent)
    reconstruction = layers.Dense(input_dim, name="reconstruction")(reconstruction_hidden)

    model = keras.Model(inputs, [room_predictions, reconstruction], name="shared_hvac_detector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"room_predictions": "mse", "reconstruction": "mse"},
        loss_weights={"room_predictions": 0.6, "reconstruction": 0.4},
        metrics={"room_predictions": [keras.metrics.MeanAbsoluteError(name="mae")]},
    )
    return model


def training_callbacks(checkpoint_path: str):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, min_delta=1e-4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
    ]


def set_reproducible(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

