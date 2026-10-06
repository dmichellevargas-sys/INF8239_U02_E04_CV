"""Auditoría de Fashion-MNIST: forma, balance, intensidades y duplicados.

Uso:  uv run python scripts/audit_data.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inf8239_u02_cv.config import CLASS_NAMES, CLASS_NAMES_ES, REPORTS
from inf8239_u02_cv.data import image_hashes, load_fashion_mnist, near_duplicates


def run() -> dict:
    REPORTS.mkdir(exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    h_train, h_test = image_hashes(x_train), image_hashes(x_test)
    near_mask, near_dist, near_idx = near_duplicates(x_train, x_test)
    audit = {
        "train_shape": list(x_train.shape), "test_shape": list(x_test.shape),
        "dtype": str(x_train.dtype),
        "pixel_range": [int(x_train.min()), int(x_train.max())],
        "pixel_mean": {"train": float(x_train.mean() / 255), "test": float(x_test.mean() / 255)},
        "pixel_std": {"train": float(x_train.std() / 255), "test": float(x_test.std() / 255)},
        "blank_images": int((x_train.max(axis=(1, 2)) == 0).sum() + (x_test.max(axis=(1, 2)) == 0).sum()),
        "class_counts": {
            "train": {CLASS_NAMES[i]: int(c) for i, c in enumerate(np.bincount(y_train, minlength=10))},
            "test": {CLASS_NAMES[i]: int(c) for i, c in enumerate(np.bincount(y_test, minlength=10))},
        },
        "exact_duplicates": {
            "within_train": int(len(h_train) - len(set(h_train))),
            "within_test": int(len(h_test) - len(set(h_test))),
            "test_in_train": int(np.isin(h_test, h_train).sum()),
        },
        "near_duplicates_test_vs_train": {
            "threshold_rms_14x14": 0.02,
            "count": int(near_mask.sum()),
            "with_different_label": int((y_train[near_idx][near_mask] != y_test[near_mask]).sum()),
        },
    }
    (REPORTS / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Muestras por clase e imagen promedio
    fig, axes = plt.subplots(10, 7, figsize=(8, 12))
    rng = np.random.default_rng(42)
    for cls in range(10):
        idx = rng.choice(np.flatnonzero(y_train == cls), 6, replace=False)
        for col, i in enumerate(idx):
            axes[cls, col].imshow(x_train[i], cmap="gray_r")
        axes[cls, 6].imshow(x_train[y_train == cls].mean(axis=0), cmap="magma")
        axes[cls, 0].set_ylabel(CLASS_NAMES_ES[cls], rotation=0, ha="right", va="center", fontsize=9)
        for ax in axes[cls]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 6].set_title("Promedio", fontsize=9)
    fig.suptitle("Fashion-MNIST: 6 ejemplos y la imagen promedio por clase (entrenamiento)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_audit_samples.png", dpi=150)
    plt.close(fig)

    # Casi duplicados encontrados
    order = np.flatnonzero(near_mask)[np.argsort(near_dist[near_mask])]
    if len(order):
        fig, axes = plt.subplots(2, len(order), figsize=(1.4 * len(order), 3.2), squeeze=False)
        for col, i in enumerate(order):
            axes[0, col].imshow(x_test[i], cmap="gray_r")
            axes[0, col].set_title(f"test {i}\n{CLASS_NAMES_ES[y_test[i]]}", fontsize=7)
            axes[1, col].imshow(x_train[near_idx[i]], cmap="gray_r")
            axes[1, col].set_title(f"train {near_idx[i]}\nd={near_dist[i]:.3f}", fontsize=7)
            for ax in axes[:, col]:
                ax.axis("off")
        fig.suptitle("Casi duplicados entre prueba (arriba) y entrenamiento (abajo)")
        fig.tight_layout()
        fig.savefig(REPORTS / "fig_audit_near_duplicates.png", dpi=150)
        plt.close(fig)
    return audit


def main() -> None:
    audit = run()
    print(json.dumps({k: audit[k] for k in ("train_shape", "test_shape", "pixel_range",
                                            "blank_images", "exact_duplicates",
                                            "near_duplicates_test_vs_train")}, indent=2))
    print("Clases (train):", set(audit["class_counts"]["train"].values()),
          "| (test):", set(audit["class_counts"]["test"].values()))
    print("Artefactos: reports/audit_summary.json, fig_audit_samples.png, fig_audit_near_duplicates.png")


if __name__ == "__main__":
    main()
