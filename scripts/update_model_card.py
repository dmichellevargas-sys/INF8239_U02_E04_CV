"""Rellena las tablas de MODEL_CARD.md con los resultados de reports/.

Mantiene la Model Card sincronizada con la última ejecución (métricas, costo y hardware).
Uso:  uv run python scripts/update_model_card.py
"""
from __future__ import annotations

import json
import re

import pandas as pd

from inf8239_u02_cv.config import CLASS_NAMES_ES, REPORTS, ROOT

CARD = ROOT / "MODEL_CARD.md"
START, END = "<!-- AUTO:START -->", "<!-- AUTO:END -->"
NAMES = {"dense": "Densa (línea base)", "cnn_gap": "CNN base (pooling global)",
         "cnn": "CNN (aplanado + densa)"}


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def build() -> str:
    res = json.loads((REPORTS / "cv_results.json").read_text(encoding="utf-8"))
    summ = pd.read_csv(REPORTS / "cv_metrics.csv")
    pc = pd.read_csv(REPORTS / "per_class_metrics.csv")
    rob = pd.read_csv(REPORTS / "robustness.csv")
    sel = res["selected_model"]
    env = res["environment"]

    metrics = pd.DataFrame({
        "Modelo": [NAMES[m] + (" **(seleccionado)**" if m == sel else "") for m in summ.model],
        "F1 validación": summ.valid_f1_macro.map("{:.4f}".format),
        "Exactitud prueba": summ.test_accuracy.map("{:.4f}".format),
        "F1 macro prueba [IC 95 %]": [f"{a:.4f} [{b:.4f}, {c:.4f}]" for a, b, c in
                                      zip(summ.test_f1_macro, summ.test_f1_ci95_low,
                                          summ.test_f1_ci95_high)],
    })
    cost = pd.DataFrame({
        "Modelo": [NAMES[m] for m in summ.model],
        "Parámetros": summ.parameters.map(lambda v: f"{v:,}".replace(",", " ")),
        "Archivo .keras (KB)": summ.size_kb.map(lambda v: f"{v:,.0f}".replace(",", " ")),
        "Épocas (mejor)": [f"{e} ({b})" for e, b in zip(summ.epochs_run, summ.best_epoch)],
        "Entrenamiento (s)": summ.train_seconds.map("{:.1f}".format),
        "Inferencia lote (ms/img)": summ.batch_ms_per_image.map("{:.3f}".format),
        "Latencia 1 imagen (ms)": summ.single_image_latency_ms.map("{:.1f}".format),
    })
    per_class = pc[pc.model == sel].copy()
    per_class["Clase"] = [CLASS_NAMES_ES[i] for i in per_class.class_id]
    per_class = per_class[["Clase", "precision", "recall", "f1", "support"]].rename(columns={
        "precision": "Precisión", "recall": "Sensibilidad", "f1": "F1", "support": "Soporte"})
    for c in ("Precisión", "Sensibilidad", "F1"):
        per_class[c] = per_class[c].map("{:.3f}".format)
    robust = rob.rename(columns={"scenario": "Escenario", "accuracy": "Exactitud",
                                 "f1_macro": "F1 macro"})
    robust["Exactitud"] = robust["Exactitud"].map("{:.4f}".format)
    robust["F1 macro"] = robust["F1 macro"].map("{:.4f}".format)
    split = res["split"]
    hw = (f"{env['platform']} · procesador {env['processor']} · {env['cpu_count']} hilos · "
          f"GPU: {env['gpu'] or 'ninguna'} · TensorFlow {env['tensorflow']} · Python {env['python']}")
    return "\n\n".join([
        START,
        ("_Sección generada por `scripts/update_model_card.py` a partir de `reports/` "
        "(última ejecución)._"),
        (f"**Partición usada:** entrenamiento {split['train']} · validación {split['valid']} · "
        f"prueba {split['test']} (se excluyeron {split['removed_near_duplicates_valid']} y "
        f"{split['removed_near_duplicates_test']} casi duplicados de validación y prueba). "
        f"Huella de índices de entrenamiento: `{split['train_index_sha256'][:16]}…`."),
        "### Métricas globales", md_table(metrics),
        (f"McNemar densa vs. seleccionado: p = {res['mcnemar_dense_vs_selected']['p_value']:.2e}; "
        f"CNN base vs. CNN: p = {res['mcnemar_cnn_gap_vs_cnn']['p_value']:.2e}."),
        "### Métricas por clase del modelo seleccionado (prueba)", md_table(per_class),
        "### Costo", f"**Hardware de medición:** {hw}", md_table(cost),
        "### Robustez ante perturbaciones (prueba)", md_table(robust),
        END,
    ])


def main() -> None:
    text = CARD.read_text(encoding="utf-8")
    auto = build()
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: auto, text, flags=re.DOTALL)
    else:
        text += "\n" + auto + "\n"
    CARD.write_text(text, encoding="utf-8")
    print("MODEL_CARD.md actualizado")


if __name__ == "__main__":
    main()
