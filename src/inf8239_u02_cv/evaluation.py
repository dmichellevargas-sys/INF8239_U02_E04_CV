"""Métricas globales y por clase, intervalos, comparación pareada y confusiones."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def global_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def per_class_table(y_true, y_pred, class_names) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class_id": labels, "class": class_names, "precision": p,
                         "recall": r, "f1": f, "support": s})


def bootstrap_f1_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [f1_score(y_true[i], y_pred[i], average="macro")
              for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def top_confusions(y_true, y_pred, class_names, k: int = 10) -> pd.DataFrame:
    """Pares (real -> predicha) más frecuentes fuera de la diagonal."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.fill_diagonal(cm, 0)
    rows = [{"true_id": i, "pred_id": j, "true": class_names[i], "pred": class_names[j],
             "count": int(cm[i, j]),
             "share_of_true_class": float(cm[i, j] / max(np.sum(np.asarray(y_true) == i), 1))}
            for i in range(cm.shape[0]) for j in range(cm.shape[1]) if cm[i, j] > 0]
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k).reset_index(drop=True)


def choose_model(valid_scores: dict, costs: dict, tolerance: float = 0.01) -> str:
    """Mejor F1 macro en validación; si otro modelo queda a menos de `tolerance`,
    se prefiere el de menor costo de entrenamiento."""
    best = max(valid_scores, key=valid_scores.get)
    eligible = [m for m, s in valid_scores.items() if valid_scores[best] - s < tolerance]
    return min(eligible, key=lambda m: costs[m])
