# Declaración de uso de herramientas de IA · Ejercicio 04

## Herramienta
- **Claude (Anthropic)**, interfaz web claude.ai, en un proyecto de la asignatura. Octubre de 2026.

## Prompts relevantes
1. Se compartieron los proyectos base de PLN y visión con el enunciado y la rúbrica de los Ejercicios 03 y 04: *"Vamos a trabajar los archivos por separado"*.
2. Tras completar el Ejercicio 03: *"Continúa con el ejercicio 4"*.

## Qué produjo la IA
- Descarga reproducible desde GitHub con verificación SHA-256, lector de archivos IDX y auditoría (duplicados exactos y casi duplicados).
- Partición estratificada con huella de índices, entrenamiento determinista, medición de costo, curvas, métricas por clase, galería de errores, pruebas de robustez y selección de modelo.
- La CNN con capa densa, propuesta al observar que la CNN del proyecto base quedaba por debajo de la línea base.
- Pruebas, notebook, README, Dataset Card, Model Card y el script que la sincroniza con los reportes.

## Partes verificadas por la estudiante
> Marque solo lo que haya comprobado personalmente.

- [ ] Ejecuté `uv sync --extra cpu`, `uv run pytest -q` y `uv run python scripts/run_all.py` en mi equipo; la Model Card quedó actualizada con mi hardware.
- [ ] Revisé las figuras de casi duplicados y de errores y estoy de acuerdo con su interpretación.
- [ ] Comprobé que la prueba solo se usa después de elegir el modelo con validación.
- [ ] Verifiqué la licencia MIT y las referencias.
- [ ] Puedo explicar cada archivo de `src/` y `scripts/`.

## Correcciones realizadas durante el desarrollo
| Problema detectado | Corrección |
|---|---|
| La validación de la base eran las últimas 6 000 imágenes, sin estratificar. | Partición estratificada con semilla y huella SHA-256 de los índices. |
| `keras.datasets` descarga desde un servidor que no permite fijar versión. | Descarga desde el repositorio oficial en un commit fijado y verificación de hashes. |
| La CNN base (pooling global) obtuvo F1 ≈ 0.82, por debajo de la línea base densa (≈ 0.87). | Se conservó como modelo comparado y se añadió una CNN con capa densa; el cambio se documenta como hallazgo. |
| Las figuras de errores mostraban números de clase. | Se usan nombres de clase y la confianza de cada predicción. |
| Un proceso de entrenamiento se lanzó con una versión desactualizada de `models.py`. | Se detuvo, se corrigió el archivo y se volvió a entrenar desde cero. |
| Avisos de Ruff (conversión `int` innecesaria, variables sin usar). | Corregidos. |

## Responsabilidad
La estudiante conserva la responsabilidad sobre datos, código, referencias y conclusiones, y declara poder explicar cualquier fragmento entregado.
