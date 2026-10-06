"""Lectura IDX, normalización, validación, partición estratificada y duplicados."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np

from .config import RAW_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_idx(path: Path) -> np.ndarray:
    """Lee un archivo IDX comprimido (formato original de MNIST/Fashion-MNIST)."""
    with gzip.open(path, "rb") as file:
        data = file.read()
    magic = int.from_bytes(data[:4], "big")
    if data[0:2] != b"\x00\x00" or data[2] != 0x08:
        raise ValueError(f"Cabecera IDX no válida ({magic:#x}) en {path}")
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    array = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"Tamaño inconsistente en {path}")
    return array.reshape(shape)


def load_fashion_mnist(raw_dir: Path = RAW_DIR):
    """Devuelve (x_train, y_train), (x_test, y_test) como uint8, igual que keras.datasets."""
    raw_dir = Path(raw_dir)
    missing = [p for p in ("train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz")
               if not (raw_dir / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan {missing}. Ejecute: uv run python scripts/download_data.py")
    x_train = read_idx(raw_dir / "train-images-idx3-ubyte.gz")
    y_train = read_idx(raw_dir / "train-labels-idx1-ubyte.gz")
    x_test = read_idx(raw_dir / "t10k-images-idx3-ubyte.gz")
    y_test = read_idx(raw_dir / "t10k-labels-idx1-ubyte.gz")
    return (x_train, y_train), (x_test, y_test)


def normalize_images(images: np.ndarray) -> np.ndarray:
    values = images.astype("float32") / 255.0
    return values[..., None]


def validate_images(images: np.ndarray, labels: np.ndarray) -> None:
    if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
        raise ValueError(f"Forma inesperada: {images.shape}")
    if len(images) != len(labels):
        raise ValueError("Imágenes y etiquetas no coinciden")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Las imágenes deben estar normalizadas entre 0 y 1")
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError("Etiquetas fuera del rango 0-9")


def stratified_split(labels: np.ndarray, valid_fraction: float, seed: int):
    """Índices de entrenamiento y validación con la misma proporción de clases (reproducible)."""
    rng = np.random.default_rng(seed)
    train_idx, valid_idx = [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_valid = round(len(idx) * valid_fraction)
        valid_idx.append(idx[:n_valid])
        train_idx.append(idx[n_valid:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(valid_idx))


def image_hashes(images: np.ndarray) -> np.ndarray:
    """Huella exacta de cada imagen (bytes crudos)."""
    return np.array([hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest() for img in images])


def duplicates_against(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Máscara de imágenes de `candidate` idénticas a alguna de `reference`."""
    ref = set(image_hashes(reference))
    return np.array([h in ref for h in image_hashes(candidate)], dtype=bool)


def split_fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


NEAR_DUP_THRESHOLD = 0.02


def _downsample(images: np.ndarray) -> np.ndarray:
    """Promedia bloques 2x2 (28x28 -> 14x14) y escala a [0, 1]."""
    x = np.asarray(images, dtype="float32").reshape(len(images), 14, 2, 14, 2)
    return x.mean(axis=(2, 4)).reshape(len(images), -1) / 255.0


def near_duplicates(reference: np.ndarray, candidate: np.ndarray,
                    threshold: float = NEAR_DUP_THRESHOLD):
    """Casi duplicados: distancia RMS por píxel (14x14) al vecino más cercano < threshold.
    Devuelve (máscara, distancias, índice del vecino en reference)."""
    from sklearn.neighbors import NearestNeighbors

    ref, cand = _downsample(reference), _downsample(candidate)
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    dist, idx = nn.kneighbors(cand)
    rms = dist.ravel() / np.sqrt(ref.shape[1])
    return rms < threshold, rms, idx.ravel()
