# Contexto del proyecto: Deep Learning, proyecto final (EAFIT)

> Este archivo es la fuente de verdad del proyecto. Léelo completo antes de proponer o modificar código. Si algo aquí contradice una suposición tuya, gana este archivo; si falta información, pregunta antes de inventar.

## 1. Qué estamos construyendo

Proyecto final grupal (2 personas: Felipe Giraldo y Santiago Villamizar) del curso de Deep Learning. Trabajamos con videos microscópicos de espermatozoides humanos del dataset **VISEM** para estimar la **motilidad** de cada muestra (porcentaje progresiva, no progresiva e inmóvil) como apoyo al análisis de calidad seminal.

**No** estamos determinando "fertilidad": ningún dataset trae esa etiqueta y no es defendible clínicamente. No escribas código, texto ni conclusiones que afirmen diagnosticar fertilidad.

### Objetivo general (borrador)
Desarrollar un sistema de Deep Learning que, a partir de videos microscópicos de VISEM, detecte y siga espermatozoides y estime la motilidad de la muestra como apoyo al análisis de calidad seminal.

### Objetivos específicos (borrador)
1. **Detección y tracking:** entrenar un detector (YOLO) con EVISAN y evaluar su transferencia a VISEM (mAP en EVISAN + revisión manual de frames de VISEM).
2. **Características de movimiento:** extraer conteo, velocidad y linealidad de trayectorias y usarlas para estimar la motilidad (MAE/RMSE vs. valores de laboratorio, división por participante).
3. **Comparación y análisis crítico:** comparar contra una línea base directa (CNN/modelo de video sobre clips) y analizar cuándo y por qué falla cada enfoque.

Plan B si la transferencia EVISAN→VISEM falla: la línea base directa pasa a ser el resultado principal y el análisis del fallo se documenta como hallazgo.

## 2. Datasets

### VISEM (dataset principal)
- 85 participantes hombres (18+), recolectados en Noruega (2008-2013).
- Más de 35 GB de video: un video por participante, 2 a 7 min, 640×480, 50 fps, microscopio Olympus CX31 a 400x.
- Archivos de video nombrados con ID de participante, fecha de captura y descripción opcional; el final del nombre trae el código de quien evaluó el video.
- 5 CSV de datos (una fila por participante) + 1 CSV que mapea video↔participante + un archivo de texto con descripciones de columnas:
  - `semen_analysis_data` (etiquetas clínicas: motilidad, concentración, volumen, vitalidad, morfología)
  - `fatty_acids_spermatozoa`, `fatty_acids_serum`, `sex_hormones`, `participant_related_data` (edad, abstinencia, IMC)
  - `videos` (mapeo)
- **Licencia:** solo investigación y educación; prohibido uso comercial. Hay que citar: Haugen et al., "VISEM: A Multimodal Video Dataset of Human Spermatozoa", MMSys'19, https://dl.acm.org/doi/10.1145/3304109.3325814
- Pendiente de verificar: si existe una extensión con anotaciones de tracking sobre los mismos videos (VISEM-Tracking). Si existe, evita el cambio de dominio.

### EVISAN (dataset de apoyo / guía del profesor)
- 6.000 imágenes JPEG + 6.000 XML con bounding boxes (clase y posición de cada espermatozoide).
- Primeras 3.000 a 200X, últimas 3.000 a 600X. Cada 1.000 imágenes forman una secuencia de video continua.
- 2 videos de prueba independientes de las imágenes (sirven para generalización).
- Licencia CC BY 4.0. Cita: Yan, Y. (2020). EVISAN, Zenodo, https://doi.org/10.5281/zenodo.4303768
- Uso previsto: entrenar el detector. **No** tiene etiquetas clínicas.

## 3. Reglas metodológicas (no negociables)

1. **Dividir train/val/test por participante** (VISEM) y **por secuencia** (EVISAN). Nunca por frame o clip: las imágenes consecutivas son casi idénticas y se filtra información entre conjuntos.
2. Todos los clips de un mismo participante van al mismo conjunto.
3. Con solo 85 participantes: preferir validación cruzada por participante (k-fold agrupado) y reportar media y desviación, no un único número.
4. Fijar semillas (`random`, `numpy`, `torch`) y registrar versiones de librerías.
5. Extraer frames/clips **una sola vez** a disco (formato ligero) y trabajar desde ahí; no decodificar los 35 GB de video en cada época.
6. Guardar checkpoints con frecuencia y registrar métricas por época (loss y métricas de train y val).
7. Toda decisión (normalización, aumentos, hiperparámetros, arquitectura) debe quedar **justificada por escrito** en el notebook: el profesor evalúa comprensión, no solo la métrica.
8. Las curvas se **analizan**, no solo se muestran: sobreajuste, subajuste, convergencia.
9. Evaluar sobre datos no vistos en entrenamiento y discutir fortalezas, limitaciones y fallas.
10. No inventar resultados, métricas ni citas. Si algo no se ha ejecutado, decirlo.

## 4. Entorno técnico

- Equipo local: GPU NVIDIA RTX 5070 Ti, 12 GB VRAM (arquitectura Blackwell).
  - Requiere PyTorch reciente con **CUDA 12.8 o superior**. Verificar con una prueba corta de entrenamiento; si aparece "no kernel image available", la versión de PyTorch es la causa.
  - Usar precisión mixta (AMP), batch pequeño con acumulación de gradiente o clips cortos si falta memoria.
- El compañero usa Google Colab (sesiones que se cortan, disco temporal): guardar clips procesados y checkpoints en almacenamiento persistente (Drive).
- Lenguaje: Python. Framework principal: PyTorch. Detector: Ultralytics YOLO (o equivalente).
- Trabajo en notebooks, con código reutilizable extraído a `src/` cuando se repita.

## 5. Estructura de carpetas

```
proyecto-dl/
├── AGENTS.md                  # este archivo
├── data/
│   ├── raw/                   # VISEM y EVISAN originales (no se modifican, no se suben a git)
│   ├── processed/             # clips/frames extraídos
│   └── splits/                # CSV con la asignación participante→train/val/test
├── notebooks/
│   └── 00_proyecto_dl.ipynb   # notebook principal (esqueleto con las secciones del profe)
├── src/                       # utilidades reutilizables (datasets, métricas, plots)
├── models/                    # checkpoints (no a git)
└── reports/
    ├── figures/
    └── informe_final.pdf      # máximo 3 páginas
```

## 6. Entregables y fechas

| Entrega | Fecha |
| --- | --- |
| Título y objetivos (correo al profesor) | jueves 29 de octubre de 2026 (el texto del README dice viernes 30; usar el 29 por seguridad) |
| Presentación + informe final (PDF, máx. 3 páginas) | miércoles 11 de noviembre de 2026 |
| Sustentación | semanas 17 y 18 (15 min + 5 min de preguntas) |

- No se aceptan versiones posteriores a la fecha límite; no entregar equivale a 0.0.
- Peso: sustentación oral 15 %, informe 5 % (del total del curso).
- El informe debe tener: objetivos, base de datos y preprocesamiento (incluidos los splits), modelo y decisiones metodológicas, resultados (con datos no vistos), conclusiones y limitaciones.
- La presentación debe cubrir: problema y objetivos, arquitectura (con referencia y modificaciones), datos y preprocesamiento, entrenamiento (curvas, hiperparámetros, pérdida, optimizador, recursos de cómputo), resultados con muestras externas, conclusiones.
- Ambos integrantes deben poder responder sobre **cualquier** parte del proyecto.

## 7. Cómo quiero que trabajes

- Responde en español; comentarios de código y nombres de variables consistentes (todo en inglés o todo en español).
- Antes de escribir código largo, explica el enfoque en pocas líneas y espera confirmación si hay más de una opción razonable.
- Prefiere soluciones simples y legibles; esto es un proyecto de aprendizaje, así que explica el "por qué" de cada decisión de modelado.
- Cada celda de código del notebook debe tener una celda markdown previa que diga qué hace y por qué.
- No modifiques `data/raw/`. No borres checkpoints ni resultados sin preguntar.
- Si detectas una posible fuga de datos, un error metodológico o una afirmación que los resultados no respaldan, avísalo aunque no te lo haya pedido.

## 8. Estado actual

- [ ] Título y objetivos enviados al profesor
- [x] VISEM descargado y estructura verificada (`visem-dataset/`: 85 videos, 6 CSV con `;` y decimal `,`; motilidad suma 100)
- [x] EVISAN descargado (`EVISAN/`: 6.000 imágenes 1200×936, 6.000 XML con clases `black`/`white`, 2 videos `.wmv`)
- [ ] Verificado si existe VISEM-Tracking
- [ ] Splits por participante definidos y guardados
- [ ] Clips/frames extraídos
- [ ] Detector entrenado con EVISAN
- [ ] Línea base de video entrenada
- [ ] Features de movimiento + regresión de motilidad
- [ ] Informe (3 páginas) y presentación

_Actualizar esta sección al final de cada sesión de trabajo._
