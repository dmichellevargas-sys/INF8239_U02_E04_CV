import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u02_cv.models import BUILDERS, compile_model


@pytest.mark.parametrize("name", list(BUILDERS))
def test_output_is_probability_over_ten_classes(name):
    model = BUILDERS[name](tf)
    probabilities = model(np.zeros((2, 28, 28, 1), dtype="float32")).numpy()
    assert probabilities.shape == (2, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_cnn_uses_convolutions_and_baseline_does_not():
    kinds = {n: {type(layer).__name__ for layer in BUILDERS[n](tf).layers} for n in BUILDERS}
    assert "Conv2D" in kinds["cnn"] and "Conv2D" in kinds["cnn_gap"]
    assert "Conv2D" not in kinds["dense"]


def test_one_training_step_gives_finite_loss():
    model = compile_model(BUILDERS["cnn"](tf))
    x = np.random.default_rng(0).random((8, 28, 28, 1)).astype("float32")
    y = np.arange(8) % 10
    history = model.fit(x, y, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])
