# Dataset Card · Fashion-MNIST

- **Fuente:** https://github.com/zalandoresearch/fashion-mnist, commit `b2617bb6d3ffa2e429640350f613e3291e10b141`.
- **Autores:** Han Xiao, Kashif Rasul y Roland Vollgraf (Zalando Research), 2017.
- **Licencia:** MIT.
- **Archivos:** 4 archivos IDX comprimidos (imágenes y etiquetas de entrenamiento y prueba); hashes SHA-256 en `data/raw_manifest.json` y en `src/inf8239_u02_cv/config.py`.
- **Contenido:** 70 000 imágenes en escala de grises de 28×28 px (60 000 entrenamiento, 10 000 prueba) de artículos del catálogo de Zalando, con 10 clases balanceadas.

| Id | Clase | Traducción |
|---|---|---|
| 0 | T-shirt/top | Camiseta/top |
| 1 | Trouser | Pantalón |
| 2 | Pullover | Suéter |
| 3 | Dress | Vestido |
| 4 | Coat | Abrigo |
| 5 | Sandal | Sandalia |
| 6 | Shirt | Camisa |
| 7 | Sneaker | Zapatilla |
| 8 | Bag | Bolso |
| 9 | Ankle boot | Botín |

## Calidad observada (`reports/audit_summary.json`)
- Formas 60 000×28×28 y 10 000×28×28, tipo `uint8`, rango 0–255, ninguna imagen en blanco.
- 6 000 imágenes por clase en entrenamiento y 1 000 en prueba.
- Sin duplicados exactos dentro ni entre particiones.
- 10 casi duplicados de prueba respecto de entrenamiento (RMS < 0.02 tras reducir a 14×14), con la misma etiqueta; ver `reports/fig_audit_near_duplicates.png`.

## Población cubierta y limitaciones
Fotografías de producto procesadas (fondo oscuro, prenda centrada y reescalada) de un único minorista europeo. No representa fotografías reales de ropa ni contiene personas.

## Cita
Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms* (arXiv:1708.07747). arXiv.
