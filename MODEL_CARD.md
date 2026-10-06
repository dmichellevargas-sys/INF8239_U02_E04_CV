# Model Card · Clasificador de prendas Fashion-MNIST (CNN)

## Modelo y versión
- **Nombre:** `cnn_flatten` (`models/cnn.keras`), versión 1.0, octubre de 2026.
- **Tipo:** red neuronal convolucional para clasificación de imágenes en 10 clases.
- **Arquitectura:** Conv2D(32, 3×3, ReLU) → MaxPool 2×2 → Conv2D(64, 3×3, ReLU) → MaxPool 2×2 → Flatten → Dropout(0.3) → Dense(64, ReLU) → Dense(10, softmax). 220 234 parámetros.
- **Entrenamiento:** Adam (tasa por defecto), entropía cruzada categórica dispersa, lote 128, hasta 15 épocas con *early stopping* sobre la pérdida de validación (paciencia 3, se restauran los mejores pesos). Semilla 42 y operaciones deterministas de TensorFlow.
- **Alternativas evaluadas:** red densa de una capa oculta (línea base) y la CNN del proyecto base del curso, que usa *pooling* global en lugar de la capa densa.
- **Responsable:** [nombre de la estudiante], INF-8239 Ciencia de Datos II, UASD. Proyecto base: Edwin Ramón José Nolasco.

## Uso previsto
Demostración académica de clasificación de imágenes: comparar arquitecturas, analizar errores por clase y medir costo. Usuarios previstos: estudiantes y docentes del curso. Entrada: imagen en escala de grises de 28×28 px, prenda centrada sobre fondo uniforme, píxeles normalizados a [0, 1] con fondo oscuro (valor 0).

## Usos fuera de alcance
- Fotografías reales de ropa (fondos, iluminación, personas, varias prendas) o imágenes a color.
- Decisiones comerciales automáticas (catalogación, precios, inventario) sin revisión humana.
- Cualquier inferencia sobre personas (talla, género, estilo o nivel socioeconómico).
- Imágenes con la prenda desplazada, recortada o con colores invertidos (ver Robustez).

## Dataset y particiones
- **Fuente:** Fashion-MNIST (Xiao, Rasul y Vollgraf, 2017), Zalando Research, licencia MIT. Descargado de https://github.com/zalandoresearch/fashion-mnist en el commit `b2617bb6` y verificado con SHA-256 (`data/raw_manifest.json`).
- **Contenido:** 70 000 imágenes 28×28 de artículos de catálogo de Zalando; 60 000 de entrenamiento y 10 000 de prueba, 10 clases balanceadas.
- **Auditoría:** sin imágenes en blanco ni duplicados exactos; 10 imágenes de prueba son casi duplicados de imágenes de entrenamiento (distancia RMS < 0.02 en 14×14), todas con la misma etiqueta.
- **Partición:** validación = 10 % estratificado del entrenamiento oficial (semilla 42). Los casi duplicados de validación y prueba respecto del entrenamiento se excluyen. La prueba se usó una sola vez.

## Preprocesamiento
Conversión a `float32`, división por 255 y canal único (28×28×1). Sin aumento de datos. La validación de entrada (`validate_images`) rechaza formas distintas a 28×28×1, valores fuera de [0, 1] y etiquetas fuera de 0–9.

## Criterio de selección
Mayor F1 macro en validación; si dos modelos difieren en menos de 1 punto, se prefiere el de menor tiempo de entrenamiento. La CNN supera a la línea base por unos 4–5 puntos de F1 en validación, por lo que se selecciona pese a su mayor costo.

## Resultados, costo y robustez
<!-- AUTO:START -->

_Sección generada por `scripts/update_model_card.py` a partir de `reports/` (última ejecución)._

**Partición usada:** entrenamiento 54000 · validación 5993 · prueba 9991 (se excluyeron 7 y 9 casi duplicados de validación y prueba). Huella de índices de entrenamiento: `f21e3d1771bfb062…`.

### Métricas globales

| Modelo | F1 validación | Exactitud prueba | F1 macro prueba [IC 95 %] |
|---|---|---|---|
| Densa (línea base) | 0.8789 | 0.8737 | 0.8727 [0.8664, 0.8791] |
| CNN base (pooling global) | 0.8205 | 0.8208 | 0.8194 [0.8122, 0.8258] |
| CNN (aplanado + densa) **(seleccionado)** | 0.9205 | 0.9221 | 0.9213 [0.9161, 0.9270] |

McNemar densa vs. seleccionado: p = 2.35e-65; CNN base vs. CNN: p = 1.89e-176.

### Métricas por clase del modelo seleccionado (prueba)

| Clase | Precisión | Sensibilidad | F1 | Soporte |
|---|---|---|---|---|
| Camiseta/top | 0.844 | 0.925 | 0.883 | 998 |
| Pantalón | 0.998 | 0.983 | 0.990 | 997 |
| Suéter | 0.899 | 0.863 | 0.881 | 1000 |
| Vestido | 0.926 | 0.933 | 0.929 | 999 |
| Abrigo | 0.841 | 0.911 | 0.875 | 999 |
| Sandalia | 0.992 | 0.973 | 0.982 | 1000 |
| Camisa | 0.818 | 0.705 | 0.757 | 998 |
| Zapatilla | 0.959 | 0.972 | 0.965 | 1000 |
| Bolso | 0.983 | 0.984 | 0.984 | 1000 |
| Botín | 0.962 | 0.972 | 0.967 | 1000 |

### Costo

**Hardware de medición:** Linux-6.18.44-fc-v71-x86_64-with-glibc2.39 · procesador x86_64 · 1 hilos · GPU: ninguna · TensorFlow 2.21.0 · Python 3.12.13

| Modelo | Parámetros | Archivo .keras (KB) | Épocas (mejor) | Entrenamiento (s) | Inferencia lote (ms/img) | Latencia 1 imagen (ms) |
|---|---|---|---|---|---|---|
| Densa (línea base) | 50 890 | 619 | 15 (15) | 22.0 | 0.021 | 2.6 |
| CNN base (pooling global) | 19 466 | 262 | 15 (15) | 303.9 | 0.135 | 5.9 |
| CNN (aplanado + densa) | 220 234 | 2 620 | 15 (13) | 436.5 | 0.134 | 6.1 |

### Robustez ante perturbaciones (prueba)

| Escenario | Exactitud | F1 macro |
|---|---|---|
| original | 0.9221 | 0.9213 |
| ruido gaussiano σ=0.10 | 0.8693 | 0.8692 |
| desplazamiento 3 px | 0.4867 | 0.4877 |
| contraste 50 % | 0.8917 | 0.8900 |
| colores invertidos | 0.2146 | 0.1349 |

<!-- AUTO:END -->

## Interpretación de errores
- La clase **Camisa** es la más difícil (F1 ≈ 0.76): se confunde sobre todo con Camiseta/top, Abrigo y Suéter. Entre los errores más confiados hay camisas visualmente indistinguibles de camisetas, lo que sugiere ambigüedad de etiqueta en el propio dataset.
- **Suéter ↔ Abrigo** es el segundo par problemático: ambos son prendas de manga larga cuya diferencia (cierre, botones) ocupa pocos píxeles a esta resolución.
- Calzado, bolsos y pantalones superan F1 = 0.96.

## Limitaciones y riesgos
- **Sensibilidad a la posición:** un desplazamiento de 3 px reduce la exactitud a menos de la mitad; el modelo depende del centrado del catálogo.
- **Inversión de colores:** con fondo claro el modelo falla casi siempre; las imágenes reales deben convertirse al mismo formato.
- **Ruido y contraste:** degradación moderada.
- **Cambio de dominio:** el desempeño en fotografías reales no se ha medido y se espera muy inferior.
- **Sesgo del catálogo:** refleja el surtido y el estilo de un solo minorista europeo de 2017.
- **Reproducibilidad:** en otro hardware o versión de TensorFlow las métricas pueden variar en milésimas y los tiempos de forma importante.

## Supervisión y monitoreo
- Validar cada imagen de entrada (forma, rango, fondo oscuro) antes de predecir.
- Registrar la probabilidad máxima; derivar a revisión humana las predicciones con confianza < 0.80 y todas las de Camisa, Camiseta/top, Suéter y Abrigo.
- Monitorear la distribución de clases predichas y la intensidad media de píxeles; un cambio marcado indica deriva.
- Reevaluar con una muestra etiquetada del dominio real antes de cualquier uso fuera del aula.

## Consideraciones éticas
El dataset no contiene personas ni datos personales. El riesgo principal es la sobreconfianza: probabilidades cercanas a 1 en errores evidentes. No debe presentarse como sistema listo para producción.

## Referencias
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms* (arXiv:1708.07747). arXiv. https://arxiv.org/abs/1708.07747
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596
