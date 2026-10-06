import gzip

import numpy as np
import pytest

from inf8239_u02_cv.data import (
    duplicates_against,
    image_hashes,
    near_duplicates,
    normalize_images,
    read_idx,
    split_fingerprint,
    stratified_split,
    validate_images,
)


def test_normalize_adds_channel_and_scales():
    raw = np.array([[[0] * 28] * 28, [[255] * 28] * 28], dtype=np.uint8)
    result = normalize_images(raw)
    assert result.shape == (2, 28, 28, 1)
    assert result.min() == 0
    assert result.max() == 1


def test_validate_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Forma inesperada"):
        validate_images(np.zeros((2, 28, 28)), np.zeros(2))


def test_validate_rejects_unscaled_pixels():
    with pytest.raises(ValueError, match="normalizadas"):
        validate_images(np.full((1, 28, 28, 1), 255.0), np.zeros(1, dtype=int))


def test_read_idx_roundtrip(tmp_path):
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    header = b"\x00\x00\x08\x03" + b"".join(int(d).to_bytes(4, "big") for d in images.shape)
    path = tmp_path / "x.gz"
    with gzip.open(path, "wb") as f:
        f.write(header + images.tobytes())
    np.testing.assert_array_equal(read_idx(path), images)


def test_read_idx_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as f:
        f.write(b"\x01\x02\x03\x04")
    with pytest.raises(ValueError):
        read_idx(path)


def test_stratified_split_is_disjoint_balanced_and_reproducible():
    labels = np.repeat(np.arange(10), 100)
    tr, va = stratified_split(labels, 0.1, seed=42)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == len(labels)
    assert (np.bincount(labels[va]) == 10).all()
    _tr2, va2 = stratified_split(labels, 0.1, seed=42)
    assert split_fingerprint(va) == split_fingerprint(va2)


def test_exact_and_near_duplicates_are_detected():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
    near = ref[:1].astype(int) + 1
    cand = np.concatenate([ref[:1], np.clip(near, 0, 255).astype(np.uint8),
                           rng.integers(0, 256, (1, 28, 28), dtype=np.uint8)])
    assert duplicates_against(ref, cand).tolist() == [True, False, False]
    mask, _dist, idx = near_duplicates(ref, cand)
    assert mask.tolist() == [True, True, False] and idx[1] == 0
    assert len(set(image_hashes(ref))) == 20
