"""Arquitecturas comparadas.

* dense: línea base que aplana la imagen (ignora la vecindad entre píxeles).
* cnn_gap: CNN del proyecto base del curso (dos bloques convolucionales + pooling global).
* cnn: misma extracción convolucional, pero con capa densa sobre el mapa 7x7 aplanado,
  que conserva la información de posición que el pooling global descarta.
"""
from __future__ import annotations


def build_dense(tf):
    return tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="dense_baseline")


def build_cnn_gap(tf):
    return tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="small_cnn_gap")


def build_cnn(tf):
    return tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="cnn_flatten")


def compile_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


BUILDERS = {"dense": build_dense, "cnn_gap": build_cnn_gap, "cnn": build_cnn}
