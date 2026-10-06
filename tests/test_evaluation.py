import numpy as np

from inf8239_u02_cv.evaluation import (
    bootstrap_f1_ci,
    choose_model,
    global_metrics,
    mcnemar_exact,
    per_class_table,
    top_confusions,
)

NAMES = [f"c{i}" for i in range(10)]


def test_per_class_table_has_one_row_per_class():
    y = np.arange(10)
    table = per_class_table(y, y, NAMES)
    assert len(table) == 10 and (table["f1"] == 1).all()


def test_top_confusions_orders_by_count():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([6, 6, 0, 2, 1, 2])
    top = top_confusions(y_true, y_pred, NAMES)
    assert (top.loc[0, "true"], top.loc[0, "pred"], top.loc[0, "count"]) == ("c0", "c6", 2)
    assert abs(top.loc[0, "share_of_true_class"] - 2 / 3) < 1e-9


def test_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, 500)
    pred = np.where(rng.random(500) < 0.8, y, rng.integers(0, 10, 500))
    low, high = bootstrap_f1_ci(y, pred, n_boot=200)
    assert low <= global_metrics(y, pred)["f1_macro"] <= high


def test_mcnemar_identical_predictions():
    y = np.arange(10)
    assert mcnemar_exact(y, y, y)["p_value"] == 1.0


def test_choose_model_prefers_cheaper_when_difference_is_small():
    scores = {"a": 0.910, "b": 0.905, "c": 0.80}
    costs = {"a": 100, "b": 10, "c": 1}
    assert choose_model(scores, costs) == "b"
    assert choose_model({"a": 0.95, "b": 0.90}, {"a": 100, "b": 1}) == "a"
