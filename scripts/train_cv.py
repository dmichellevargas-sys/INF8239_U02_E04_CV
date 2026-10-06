"""Entrenamiento y evaluación: línea base densa vs. dos CNN (Ejercicio 04).

Protocolo declarado antes de evaluar en prueba:
  * Partición: 10 % estratificado del entrenamiento oficial como validación (semilla 42);
    la prueba oficial (10 000) se usa una sola vez, al final.
  * Fuga: se excluyen de validación y de prueba las imágenes casi duplicadas de alguna
    imagen de entrenamiento (distancia RMS 14x14 < 0.02).
  * Entrenamiento: Adam, lote 128, hasta EPOCHS épocas, EarlyStopping sobre val_loss
    (paciencia 3, se restauran los mejores pesos).
  * Selección: mayor F1 macro en VALIDACIÓN; si la diferencia es < 1 punto, el más barato.

Uso:  uv run python scripts/train_cv.py [--epochs 15]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

from inf8239_u02_cv.config import (
    CLASS_NAMES,
    CLASS_NAMES_ES,
    MODELS,
    REPORTS,
    SEED,
    VALID_FRACTION,
)
from inf8239_u02_cv.data import (
    load_fashion_mnist,
    near_duplicates,
    normalize_images,
    split_fingerprint,
    stratified_split,
    validate_images,
)
from inf8239_u02_cv.evaluation import (
    bootstrap_f1_ci,
    choose_model,
    global_metrics,
    mcnemar_exact,
    per_class_table,
    top_confusions,
)
from inf8239_u02_cv.models import BUILDERS, compile_model

LABELS = {"dense": "Densa (línea base)", "cnn_gap": "CNN base (pooling global)",
          "cnn": "CNN (aplanado + densa)"}


def set_determinism() -> None:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def prepare_data() -> dict:
    (x_all, y_all), (x_test_raw, y_test) = load_fashion_mnist()
    train_idx, valid_idx = stratified_split(y_all, VALID_FRACTION, SEED)
    x_tr_raw, y_tr = x_all[train_idx], y_all[train_idx]
    x_va_raw, y_va = x_all[valid_idx], y_all[valid_idx]
    near_va, _, _ = near_duplicates(x_tr_raw, x_va_raw)
    near_te, _, _ = near_duplicates(x_tr_raw, x_test_raw)
    x_va_raw, y_va = x_va_raw[~near_va], y_va[~near_va]
    test_kept = np.flatnonzero(~near_te)
    x_test_raw, y_test = x_test_raw[test_kept], y_test[test_kept]
    data = {
        "x_train": normalize_images(x_tr_raw), "y_train": y_tr,
        "x_valid": normalize_images(x_va_raw), "y_valid": y_va,
        "x_test": normalize_images(x_test_raw), "y_test": y_test, "test_index": test_kept,
    }
    for split in ("train", "valid", "test"):
        validate_images(data[f"x_{split}"], data[f"y_{split}"])
    split_log = {
        "train": len(y_tr), "valid": len(y_va), "test": len(y_test),
        "removed_near_duplicates_valid": int(near_va.sum()),
        "removed_near_duplicates_test": int(near_te.sum()),
        "valid_fraction": VALID_FRACTION, "seed": SEED,
        "train_index_sha256": split_fingerprint(train_idx),
        "valid_index_sha256": split_fingerprint(valid_idx),
        "class_counts": {s: np.bincount(data[f"y_{s}"], minlength=10).tolist()
                         for s in ("train", "valid", "test")},
    }
    (REPORTS / "split.json").write_text(json.dumps(split_log, indent=2), encoding="utf-8")
    return data | {"split": split_log}


def train_one(name: str, data: dict, epochs: int):
    set_determinism()
    model = compile_model(BUILDERS[name](tf))
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    start = perf_counter()
    history = model.fit(data["x_train"], data["y_train"], epochs=epochs, batch_size=128,
                        validation_data=(data["x_valid"], data["y_valid"]),
                        callbacks=[stop], verbose=2)
    train_seconds = perf_counter() - start
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.insert(0, "model", name)
    path = MODELS / f"{name}.keras"
    model.save(path)
    return model, hist, {"train_seconds": train_seconds, "epochs_run": len(hist),
                         "best_epoch": int(hist["val_loss"].idxmin()) + 1,
                         "parameters": int(model.count_params()),
                         "size_kb": path.stat().st_size / 1024}


def inference_cost(model, x) -> dict:
    model.predict(x[:256], verbose=0)  # calentamiento
    batch = []
    for _ in range(3):
        start = perf_counter()
        model.predict(x, batch_size=256, verbose=0)
        batch.append((perf_counter() - start) * 1000 / len(x))
    single = []
    sample = x[:1]
    for _ in range(30):
        start = perf_counter()
        model(sample, training=False)
        single.append((perf_counter() - start) * 1000)
    return {"batch_ms_per_image": float(np.median(batch)),
            "single_image_latency_ms": float(np.median(single))}


def plot_curves(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in history.groupby("model", sort=False):
        for metric, ax in zip(("loss", "accuracy"), axes):
            ax.plot(h["epoch"], h[metric], "--", label=f"{LABELS[name]} · entrenamiento")
            ax.plot(h["epoch"], h[f"val_{metric}"], "-o", ms=3, label=f"{LABELS[name]} · validación")
    axes[0].set(title="Pérdida (entropía cruzada)", xlabel="Época", ylabel="Pérdida")
    axes[1].set(title="Exactitud", xlabel="Época", ylabel="Exactitud")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_learning_curves.png", dpi=170)
    plt.close(fig)


def plot_confusion(y_true, y_pred, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(10)), display_labels=CLASS_NAMES_ES,
        normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set(title=f"Matriz de confusión normalizada por clase real · {LABELS[name]}",
           xlabel="Predicha", ylabel="Real")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(REPORTS / f"fig_confusion_{name}.png", dpi=170)
    plt.close(fig)


def plot_per_class(tables: dict) -> None:
    merged = pd.DataFrame({LABELS[n]: t["f1"].to_numpy() for n, t in tables.items()},
                          index=CLASS_NAMES_ES)
    ax = merged.sort_values(LABELS["cnn"]).plot.barh(figsize=(8, 5))
    ax.set(title="F1 por clase en prueba", xlabel="F1", xlim=(0.4, 1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "fig_f1_per_class.png", dpi=170)
    plt.close()


def plot_cost(summary: pd.DataFrame) -> None:
    """Desempeño en validación frente a costo: entrenamiento, tamaño y latencia."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [("train_seconds", "Tiempo de entrenamiento (s)"), ("size_kb", "Tamaño del archivo .keras (KB)"),
             ("batch_ms_per_image", "Inferencia por imagen en lote (ms)")]
    for ax, (col, label) in zip(axes, specs):
        for _, row in summary.iterrows():
            ax.scatter(row[col], row["valid_f1_macro"], s=80,
                       color="#0f7c7b" if row["selected"] else "#94a3b8")
            ax.annotate(LABELS[row["model"]], (row[col], row["valid_f1_macro"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set(xlabel=label, ylabel="F1 macro (validación)", ylim=(0.78, 0.95))
    fig.suptitle("Rendimiento y costo (verde = modelo seleccionado)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_cost_tradeoff.png", dpi=170)
    plt.close(fig)


def plot_error_gallery(x, y_true, proba, confusions: pd.DataFrame, name: str, per_pair: int = 6):
    """Errores más confiados en los tres pares de clases más confundidos."""
    pred = proba.argmax(axis=1)
    pairs = confusions.head(3)
    fig, axes = plt.subplots(len(pairs), per_pair, figsize=(1.7 * per_pair, 2.2 * len(pairs)))
    rows = []
    for r, (_, pair) in enumerate(pairs.iterrows()):
        idx = np.flatnonzero((y_true == pair.true_id) & (pred == pair.pred_id))
        idx = idx[np.argsort(-proba[idx, pair.pred_id])][:per_pair]
        for c in range(per_pair):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(idx):
                i = idx[c]
                ax.imshow(x[i].squeeze(), cmap="gray_r")
                ax.set_title(f"R: {CLASS_NAMES_ES[y_true[i]]}\nP: {CLASS_NAMES_ES[pred[i]]} "
                             f"({proba[i, pred[i]]:.2f})", fontsize=7)
                rows.append({"position_in_test": int(i), "true": CLASS_NAMES[y_true[i]],
                             "pred": CLASS_NAMES[pred[i]], "confidence": float(proba[i, pred[i]])})
    fig.suptitle(f"Errores más confiados en los 3 pares más confundidos · {LABELS[name]}")
    fig.tight_layout()
    fig.savefig(REPORTS / f"fig_errors_{name}.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REPORTS / f"errors_gallery_{name}.csv", index=False)


def robustness(model, x, y) -> pd.DataFrame:
    """Perturbaciones simples para documentar límites en la Model Card."""
    rng = np.random.default_rng(SEED)
    scenarios = {
        "original": x,
        "ruido gaussiano σ=0.10": np.clip(x + rng.normal(0, 0.10, x.shape), 0, 1).astype("float32"),
        "desplazamiento 3 px": np.roll(x, 3, axis=2),
        "contraste 50 %": (x * 0.5).astype("float32"),
        "colores invertidos": (1.0 - x).astype("float32"),
    }
    rows = []
    for label, xs in scenarios.items():
        pred = model.predict(xs, batch_size=256, verbose=0).argmax(axis=1)
        rows.append({"scenario": label, **global_metrics(y, pred)})
    return pd.DataFrame(rows)


def run(epochs: int = 15) -> dict:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    data = prepare_data()
    histories, rows, models, preds, probas, tables = [], [], {}, {}, {}, {}
    for name in BUILDERS:
        model, hist, cost = train_one(name, data, epochs)
        histories.append(hist)
        models[name] = model
        valid_pred = model.predict(data["x_valid"], verbose=0).argmax(axis=1)
        rows.append({"model": name, **cost, **inference_cost(model, data["x_test"]),
                     "valid_f1_macro": global_metrics(data["y_valid"], valid_pred)["f1_macro"]})
    summary = pd.DataFrame(rows)
    selected = choose_model(dict(zip(summary.model, summary.valid_f1_macro)),
                            dict(zip(summary.model, summary.train_seconds)))

    # Evaluación final en prueba (una sola vez)
    for name, model in models.items():
        probas[name] = model.predict(data["x_test"], verbose=0)
        preds[name] = probas[name].argmax(axis=1)
        m = global_metrics(data["y_test"], preds[name])
        low, high = bootstrap_f1_ci(data["y_test"], preds[name], seed=SEED)
        summary.loc[summary.model == name, ["test_accuracy", "test_f1_macro",
                                            "test_f1_ci95_low", "test_f1_ci95_high"]] = \
            [m["accuracy"], m["f1_macro"], low, high]
        tables[name] = per_class_table(data["y_test"], preds[name], CLASS_NAMES)
        tables[name].insert(0, "model", name)
        plot_confusion(data["y_test"], preds[name], name)
    summary["selected"] = summary["model"] == selected
    summary.to_csv(REPORTS / "cv_metrics.csv", index=False)
    plot_cost(summary)
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(REPORTS / "training_history.csv", index=False)
    pd.concat(tables.values()).to_csv(REPORTS / "per_class_metrics.csv", index=False)
    plot_curves(history)
    plot_per_class(tables)

    confusions = top_confusions(data["y_test"], preds[selected], CLASS_NAMES_ES)
    confusions.to_csv(REPORTS / "top_confusions.csv", index=False)
    plot_error_gallery(data["x_test"], data["y_test"], probas[selected], confusions, selected)
    robust = robustness(models[selected], data["x_test"], data["y_test"])
    robust.to_csv(REPORTS / "robustness.csv", index=False)

    results = {
        "selected_model": selected,
        "selection_rule": "mayor F1 macro en validación; diferencia < 0.01 -> menor tiempo de entrenamiento",
        "split": data["split"],
        "mcnemar_dense_vs_selected": mcnemar_exact(data["y_test"], preds["dense"], preds[selected]),
        "mcnemar_cnn_gap_vs_cnn": mcnemar_exact(data["y_test"], preds["cnn_gap"], preds["cnn"]),
        "errors_in_test_selected": int((preds[selected] != data["y_test"]).sum()),
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "tensorflow": tf.__version__,
            "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
            "scikit_learn": sklearn.__version__, "numpy": np.__version__,
        },
    }
    (REPORTS / "cv_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return results | {"summary": summary, "per_class": pd.concat(tables.values()),
                      "confusions": confusions, "robustness": robust, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    r = run(args.epochs)
    cols = ["model", "parameters", "size_kb", "epochs_run", "best_epoch", "train_seconds",
            "batch_ms_per_image", "single_image_latency_ms", "valid_f1_macro", "test_f1_macro",
            "test_f1_ci95_low", "test_f1_ci95_high", "selected"]
    print("Partición:", {k: r["split"][k] for k in ("train", "valid", "test",
                                                     "removed_near_duplicates_test")})
    print(r["summary"][cols].round(4).to_string(index=False))
    print("Seleccionado:", r["selected_model"])
    print("McNemar densa vs seleccionado:", r["mcnemar_dense_vs_selected"])
    print("McNemar cnn_gap vs cnn:", r["mcnemar_cnn_gap_vs_cnn"])
    sel = r["per_class"][r["per_class"].model == r["selected_model"]]
    print(sel[["class", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print("Pares más confundidos:\n", r["confusions"].head(5).round(3).to_string(index=False))
    print("Robustez:\n", r["robustness"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
