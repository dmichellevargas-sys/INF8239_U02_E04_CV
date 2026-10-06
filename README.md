# INF-8239 · Ejercicio 04 · Clasificador visual con CNN y Model Card

Clasificación de prendas en 10 clases con **Fashion-MNIST** (MIT): línea base densa frente a dos CNN,
con auditoría, partición reproducible, métricas por clase, curvas, análisis de errores, costo y Model Card.
Proyecto individual de la Unidad 02. Proyecto base del curso: Edwin Ramón José Nolasco.

## Requisitos
- [uv](https://docs.astral.sh/uv/) ≥ 0.5 (instala Python 3.12).
- Internet la primera vez (≈ 30 MB desde GitHub).
- CPU suficiente. En una CPU de portátil el entrenamiento completo tarda entre 5 y 15 minutos;
  con GPU NVIDIA (WSL2/Linux) mucho menos.

## Ejecución (uv, recomendada)
```bash
uv python install 3.12
uv sync --extra cpu                      # GPU en WSL2/Linux: uv sync --extra gpu
uv run python scripts/check_runtime.py   # versiones y GPU detectada
uv run pytest -q                         # 17 pruebas
uv run python scripts/run_all.py         # descarga -> auditoría -> entrenamiento -> Model Card
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/E04_cnn_model_card.ipynb
```
Pasos individuales: `scripts/download_data.py`, `scripts/audit_data.py`,
`scripts/train_cv.py --epochs 15` y `scripts/update_model_card.py`.

## Ejecución alternativa (pip)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
$env:PYTHONPATH="src;."             # macOS/Linux: export PYTHONPATH=src:.
python -m pytest -q
python scripts/run_all.py
```

## Estructura
```
MODEL_CARD.md               ficha del modelo (tablas actualizadas automáticamente)
data/raw_manifest.json      URLs, commit y SHA-256 de los archivos IDX (versionado)
docs/                       DATASET_CARD y declaración de uso de IA
notebooks/                  notebook ejecutado
reports/                    métricas, historial, partición y figuras
scripts/                    descarga, auditoría, entrenamiento, Model Card, run_all
src/inf8239_u02_cv/         config, data, models, evaluation
tests/                      pruebas de datos, modelos y evaluación
```

## Decisiones principales
| Decisión | Razón |
|---|---|
| Descarga desde GitHub en commit fijado, con SHA-256 | Misma fuente en cualquier equipo; no depende de la caché de Keras |
| Validación = 10 % estratificado del entrenamiento (semilla 42) | Clases balanceadas en validación; la base usaba las últimas 6 000 imágenes |
| Exclusión de casi duplicados de validación y prueba | 10 imágenes de prueba eran casi idénticas a imágenes de entrenamiento |
| Tres modelos: densa, CNN base (pooling global) y CNN con capa densa | Medir el aporte de la convolución y el efecto de la cabeza del modelo |
| *Early stopping* sobre `val_loss` y operaciones deterministas | Evitar sobreajuste y repetir resultados |
| Selección por F1 macro en validación con tolerancia de 1 punto a favor del más barato | La prueba se usa una sola vez; el costo cuenta en empates |
| Model Card sincronizada con `reports/` | Las cifras del documento siempre coinciden con la ejecución |

## Resultados (CPU, semilla 42)
| Modelo | Parámetros | F1 validación | F1 prueba [IC 95 %] | Entrenamiento (s)* |
|---|---|---|---|---|
| Densa (línea base) | 50 890 | 0.879 | 0.873 [0.866, 0.879] | 22 |
| CNN base (pooling global) | 19 466 | 0.821 | 0.819 [0.812, 0.826] | 304 |
| **CNN (aplanado + densa), seleccionada** | 220 234 | 0.921 | 0.921 [0.916, 0.927] | 437 |

\*Medido en una CPU de 1 hilo (ver `reports/cv_results.json`); depende del hardware. Dos ejecuciones completas dieron métricas idénticas.
La clase más difícil es Camisa (F1 ≈ 0.76). Un desplazamiento de 3 px reduce la exactitud a ≈ 0.49.
Detalle completo en `MODEL_CARD.md` y en el notebook.

## Artefactos
`reports/cv_metrics.csv`, `per_class_metrics.csv`, `training_history.csv`, `top_confusions.csv`,
`robustness.csv`, `errors_gallery_cnn.csv`, `split.json`, `audit_summary.json`, `cv_results.json`
y figuras `fig_*.png`. Los modelos `models/*.keras` se regeneran y no se versionan.

## Git
```bash
git init
git add pyproject.toml uv.lock requirements.txt .gitignore .python-version
git commit -m "chore: configure reproducible CV environment"
git add src/inf8239_u02_cv/config.py src/inf8239_u02_cv/data.py src/inf8239_u02_cv/__init__.py scripts/download_data.py scripts/audit_data.py data/raw_manifest.json docs/DATASET_CARD.md tests/test_data.py
git commit -m "feat: pinned Fashion-MNIST download, audit and stratified split"
git add src scripts tests
git commit -m "feat: train dense baseline and two CNNs with cost measurement"
git add reports notebooks
git commit -m "docs: add executed notebook and reports"
git add MODEL_CARD.md README.md docs/AI_USE.md
git commit -m "docs: model card, README and AI use statement"
git tag u02-ejercicio04
```

## Uso de IA
Ver `docs/AI_USE.md`.

## Referencias
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms* (arXiv:1708.07747). arXiv. https://arxiv.org/abs/1708.07747
