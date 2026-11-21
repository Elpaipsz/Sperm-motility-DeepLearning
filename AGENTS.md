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
│   └── estimacion_motilidad_visem.ipynb   # notebook principal (esqueleto con las secciones del profe)
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

- [x] Título y objetivos enviados al profesor (texto exacto en la sección 1 del notebook). Lo enviado: O1 = detector YOLO con EVISAN + transferencia a VISEM con mAP en EVISAN y VISEM-Tracking **y calidad del seguimiento de trayectorias**; O2 = características de movimiento (conteo, velocidad, linealidad) → MAE/RMSE con CV agrupada por participante; O3 = comparación con línea base directa y análisis de fallos. El informe debe cubrir exactamente eso
- [x] Repo privado en GitHub creado (`Sperm-motily-DeepLearning`); Drive compartido para clips/checkpoints/logs definido en la conversación, aún por crear
- [x] VISEM descargado y estructura verificada (`visem-dataset/`: 85 videos, 6 CSV con `;` y decimal `,`; motilidad suma 100)
- [x] EVISAN descargado (`EVISAN/`: 6.000 imágenes 1200×936, 6.000 XML con clases `black`/`white`, 2 videos `.wmv`)
- [x] Verificado si existe VISEM-Tracking: **existe** (20 videos de 30 s con cajas e IDs de trayectoria, CC BY 4.0, https://zenodo.org/record/7293726). **Descargado** en `VISEM-Tracking/` (ZIP íntegro 6,3 GB, extraído en `VISEM-Tracking/extracted/`). Cubre los participantes 11, 12, 13, 14, 15, 19, 21, 22, 23, 24, 29, 30, 35, 36, 38, 47, 52, 54, 60 y 82: ~1.470 frames de 30 s cada uno, etiquetas YOLO (`labels/`) y con ID de trayectoria (`labels_ftid/`). Clases: 0 normal (612.377 cajas), 1 clúster (21.846), 2 pinhead (22.112); cajas de ~23×23 px en 640×480. **Sesgo:** esos 20 participantes tienen motilidad progresiva media 25,9 % (máx. 58 %) frente a 42,7 % (máx. 76 %) en los 85. Riesgo de fuga si el detector se entrena con participantes que luego caen en test
- [x] Splits por participante definidos y guardados (`data/splits/folds.csv`, generado por `src/make_splits.py`, semilla 42): test fijo de 15 (4 con cajas: 13, 14, 35, 60) + 5 folds agrupados sobre 70; detector entrena con folds 1-4 y valida con fold 0. **No regenerar** una vez empiecen los experimentos. Limitación: test con 9/15 videos evaluados por JMA
- [x] Clips/frames extraídos (`data/processed/visem_gray/`, 4 ventanas de 30 s por participante; YOLO en `evisan_yolo/` y `visem_yolo/`)
- [x] Detectores entrenados (A solo EVISAN, B A→VISEM, C solo VISEM, D A→VISEM a 960 px). mAP@0.5 en VISEM-test: A 0,264 / B 0,470 / C 0,458 / D 0,492 (EVISAN-test: A 0,886). Elegido **D** por MOTA/IDF1 en validación (0,294/0,511 vs B 0,056/0,451). Tracking de D en test (una sola evaluación, 4 participantes): MOTA 0,298, IDF1 0,559. Transferencia directa EVISAN→VISEM (A) falla: es hallazgo
- [x] Línea base de video (CNN 3D, 5 folds, última época): MAE CV 9,76 / test 7,11 pp. Se están entrenando 2 semillas más (7 y 123) para un ensamble; si se usa el ensamble, decirlo en el informe
- [x] Features de movimiento (D, 85 videos) + regresión de motilidad: Ridge elegido por CV, MAE CV 7,97 ± 1,20 / test 6,81 pp. Referencia trivial (media): CV 12,80 / test 13,22. Diferencia entre Ridge y CNN en test no es concluyente (15 participantes)
- [x] Notebook principal `estimacion_motilidad_visem.ipynb` ejecutado con secciones 2.3–5.5 (curvas, tablas, figuras en `reports/figures/`). Pendiente: redactar hallazgos de 5.5 y la sección 6 (conclusiones) cuando terminen los experimentos 01/02. Hallazgos ya medidos: ambos métodos fallan en los mismos participantes (correlación del error 0,69; peores: 7, 9, 35, 22, 11, 12); en los 13 videos con nota de calidad (drift/debris) la CNN 3D falla mucho más que Ridge (16,4 vs 11,4 pp); ambos subestiman motilidad alta y sobreestiman la baja
- [x] Experimentos de DL ejecutados (3 semillas c/u, mismos folds; notebooks 01 y 02 con análisis): ResNet18 afinada CV 9,00 / test 8,13; sonda lineal 10,65 / 9,98; CNN 3D regularizada 9,04 / 7,82; CNN 3D base 9,71 / 7,00. Diferencias pareadas sobre los 85 participantes (bootstrap): ninguna variante DL se distingue de la CNN 3D base; **todas son significativamente peores que Ridge sobre trayectorias** (+1,7 a +2,1 pp). Nota: afinar layer4 NO reduce parámetros libres (29× más que la CNN 3D)
- [x] **Sistema final: detector B + Ridge** (CV 7,31 ± 1,16, test 6,33; con D: 7,97 / 6,81). D se eligió antes por seguimiento en validación (MOTA 0,29 vs 0,06) pero la diferencia venía de los participantes 19 y 23; en test B y D empatan en seguimiento (MOTA 0,30 ambos) y B da mejor motilidad (diferencia pareada 0,63 pp, IC95 [0,29, 1,01]). B se eligió por CV, no por test; es una decisión posterior a ver resultados y hay que decirlo en el informe
- [x] Detector A rastreado en los 85 videos + Ridge: MAE CV 8,10 / test 7,20 (B 7,31 / 6,33; D 7,97 / 6,81); B mejor que A en 0,81 pp [0,02; 1,62] y que D en 0,63 [0,29; 1,01]: un detector con MOTA −0,82 conserva casi toda la señal. Cuaderno `03_ablaciones_trayectorias.ipynb` ejecutado con análisis: familias (linealidad y casi quietas casi bastan; información redundante), 30 s de video bastan (7,26 vs 7,31 con 120 s), la corrección de deriva NO explica el fallo de la CNN en videos con nota (hipótesis no respaldada)
- [x] Red de conjuntos (Deep Sets, `src/track_set_net.py`, 1.478 parámetros, sin ajuste de hiperparámetros, 3 semillas, detector B): CV 7,18 ± 1,39 / test 6,45; empata con Ridge (+0,09 [−0,38; 0,53]) y supera a la CNN 3D (+2,19 [1,00; 3,52]); error correlacionado 0,90 con Ridge. Figuras de casos listas en `reports/figures/casos/`
- [x] Cuaderno `04_mejoras_dl.ipynb` ejecutado: (1) no progresiva: habilidad 0,11–0,15 sobre la media, CNN 3D −0,07; imponer U invertida no mejora; (2) CV anidada de la red de conjuntos: 7,22 vs 7,21 por defecto (+0,08 [−0,23; 0,40]); (3) r3d_18 preentrenado (24,9 M params entrenables): CV 9,59 / test 7,18, peor que Ridge +2,03 [1,00; 3,11], sobreajusta (no se distingue de la CNN 3D base); (4) incertidumbre: desacuerdo entre modelos no predice el error (Spearman 0,13–0,15) pero marca videos con nota (6,3 vs 2,2); intervalos conformales 90 % ±14–19 pp, cobertura test 0,93. La red de conjuntos tiene 1.478 parámetros (corregido)
- [x] Revisión contra la guía original del profesor (7-nov): faltaban (a) evaluación externa con los **2 videos de prueba de EVISAN** (solo estaban mencionados), (b) análisis del error frente a la **concentración**, (c) una demostración **visual** de las detecciones. Cubierto en `estimacion_motilidad_visem.ipynb` (secciones 5.1.1, 5.1.2 y 5.5.1; figuras `demo_*.png` por `src/demo_figures.py`). Hallazgos: A marca residuo como espermatozoide en VISEM (13 detectados vs 3 anotados en el participante 14); B/D omiten células con halo brillante (29 y 23 de 33 en el 35); detecciones por frame de B correlacionan 0,82 (Spearman) con la concentración de laboratorio; error de motilidad mayor con baja concentración (Ridge 8,5 pp vs 5,7 en el tercio alto; Spearman −0,23, p 0,036). Los videos de EVISAN no tienen anotaciones: la evaluación es solo cualitativa (A detecta ~8,5 células por frame). `test_600x` es de 2448×2048 y se reescala a 1200 px de ancho. **El cuaderno 00 es el entregable; 01–04 son anexos** (hay un mapa de cuadernos en su sección 1)
- [x] Prueba de características CASA extra (`src/extra_features.py`: ALH, BCF, giro medio, trayectorias en el sitio): no mejora el MAE total con Ridge (7,35 vs 7,31); la no progresiva baja a 6,4–6,6 con gradient boosting, pero es el mejor de 6 combinaciones sin comparación pareada ni CV anidada: **no citar como resultado**; no está en ningún cuaderno
- [x] Detector E (YOLO11s; Anexo F) **no se adopta** (criterio fijado antes: motilidad en CV con 30 s de video). mAP@0,5 VISEM-test 0,448 (B 0,470, D 0,492); E1 solo EVISAN 0,877 EVISAN-test / 0,212 VISEM-test; mejor seguimiento en validación (MOTA 0,304, IDF1 0,530) pero peor motilidad: CV 8,04 vs 7,26 de B (ambos 30 s), +0,77 pp [0,35; 1,23]; test (mirado después) 6,20 vs 5,97. Lectura: el límite del detector está en los datos anotados, no en la capacidad; MOTA en 4 participantes no predice la tarea final (2ª vez)
- [x] Herramienta de uso: `src/final_system.py` entrena y guarda en `models/final/` (Ridge + red de conjuntos ×3 con los 70, intervalo conformal 80/90 %, umbral de pocas trayectorias = 1.365) y `src/predict_video.py <video>` devuelve estimación + intervalo + aviso (con aviso, error de CV 9,46 vs 5,68; IC [1,66; 5,98]). Demo en sección 5.9: participante 13 acierta, 35 falla fuera del intervalo (ambos con aviso). El Ridge reproduce exactamente el test del protocolo; la red difiere ~1,5 pp (entrenada con los 70 en lugar de por fold)
- [x] Informe `reports/informe_final.docx` y `.pdf`: 3 páginas (verificado con Word); describe el sistema final B + Ridge/red, el detector E y la herramienta
- [x] **Un solo notebook** (decisión 7-nov): `estimacion_motilidad_visem.ipynb` (150 celdas tras la revisión del 11-nov, ejecutado, 0 errores) sigue la guía del profesor (secciones 1–7) e incluye al final los Anexos A–E (antes cuadernos 01–04 + características CASA extra); `01`–`04` siguen en disco sin tocar y se pueden archivar. Sección 5.8 = **historial** de 11 pasos (tabla y figura calculadas). Nuevas piezas: línea base por reglas (5.3.1, CV 9,50 / test 9,21; ≈ CNN 3D, +2,31 pp peor que Ridge), video anotado (5.1.3, `reports/demo/`, `src/demo_video.py`), evaluación con los 2 videos de EVISAN (5.1.2) y concentración (5.5.1)
- [x] **Sistema final = detector B + promedio de Ridge y red de conjuntos** (CV 7,11 / test 6,32). Elegido por CV, no por test; la diferencia con Ridge solo (+0,17 [−0,06; 0,39]) no es distinguible. Anexo E: características CASA extra (ALH, BCF, giro) NO mejoran (R1 7,35; GB 7,35 CV / 7,53 test; no progresiva +0,47 [−0,35; 1,32])
- [x] Revisadas las 3 referencias de `reference-notebooks/` (ninguna sirve como comparación: YOLO sin entrenar y 1 video; dataset de 240 imágenes; clasificador fértil/infértil con accuracy 1,0 = fuga). **`sperm-detection-project.ipynb` contiene una clave de API de Roboflow ajena: no usarla ni subir esa carpeta a git**
- [x] **Comparación con referencias y literatura** (sección 5.10): punya-tat replicado sobre nuestras trayectorias B da MAE 18,9 (120 s) / 21,1 (6 s), peor que la media (12,87); en su video, 41,3. Literatura sobre VISEM (3 folds MediaEval, sin test aparte, mejor de varias configuraciones): Thambawita et al. 2019 (ResNet-34 + flujo óptico) **8,83** (verificado en el PDF: folds 8,61/7,87/9,99); motilitAI (Ottl et al. 2022, Trackpy + BoW + SVR) **7,31** (verificado en el resumen). Mismo patrón que nosotros (video ≈ 9, trayectorias ≈ 7,3). Réplica de motilitAI con config. fija (`src/bow_svr.py`, k = 100, SVR lineal): CV 8,93 / test 8,15, peor que Ridge (+1,66 [0,77; 2,60]). R² del sistema final (85): 0,75 / 0,19 / 0,73 (motilitAI 0,74 / 0,26 / 0,66 según un resumen del artículo, por confirmar)
- [x] **Clasificación con límites OMS 2010** (sección 5.11): progresiva < 32 % (26/85): accuracy 0,88, balanced 0,85, F1 0,80 (CNN 3D 0,87 / 0,85 / 0,79; trivial 0,69); motilidad total < 40 % solo 6 casos (sistema detecta 3, CNN 0). Error ≤ 10 pp en 72 / 74 / 78 % de los participantes. **No es diagnóstico de fertilidad**
- [x] Historial resumido para la presentación: `reports/figures/resumen_presentacion.png` (4 hitos + líneas de la literatura)
- [x] **Presentación y guion (8-nov)**: `reports/presentacion.html` (18 diapositivas, un solo archivo sin internet; flechas, N = notas del orador, F = pantalla completa) y `reports/presentacion.pdf` (respaldo, 18 páginas 16:9); `reports/guion_presentacion.docx` (tiempos 14 min 20 s y reparto sugerido, texto por diapositiva, glosario de ~28 conceptos, 14 preguntas probables, cifras a mano). Se regeneran con scripts del scratchpad de la sesión (no están en el repo): si se cambia una cifra hay que editar el HTML. `.gitignore` reordenado: ignora `reference-notebooks/` (clave de API ajena), `reports/demo/`, `weights/`, `*.joblib`, secretos; se suben ~17 MB. `requirements.txt` regenerado (faltaba motmetrics)
- [x] **Revisión final contra el correo del profesor (11-nov, día de entrega)**. Notebook: se corrigió un error del script de reconstrucción que había duplicado el «Mapa del documento» 8 veces en la celda 5; se añadió la justificación de relevancia del problema (1), recursos de cómputo completos (CPU Intel Core Ultra 9 275HX, 31,4 GB RAM, RTX 5070 Ti Laptop 12 GB, Windows 11, Python 3.13.14), filas de la red de conjuntos, Ridge y detector E en la tabla de entrenamiento y las curvas de la red de conjuntos con su análisis (4); dos celdas de código sin texto previo; menciones residuales a «cuadernos 01–04»; fecha. 150 celdas.
- [x] **Presentación rehecha (11-nov)**: `reports/presentacion.html` y `.pdf` de 22 láminas, **sin animaciones** (portada estática con un frame real anotado). Añadidas láminas de preprocesamiento (normalización y aumentos justificados), cómo funcionan YOLO / red de conjuntos / CNN 3D, curvas del detector, videos externos de EVISAN y «¿responde al problema?» en conclusiones. Guion (`reports/guion_presentacion.docx`) actualizado a 22 láminas, 14 min 45 s (Felipe ~7:40, Santiago ~7:05). Informe: añadidos problema y cómputo completo, sigue en 3 páginas
- [x] **Versión profesional y entrega en .zip (11-nov)**: el cuaderno se renombró `00_proyecto_dl.ipynb` → `estimacion_motilidad_visem.ipynb` (el correo del profesor no exige notebook: solo presentación e informe en PDF) y se quitaron las notas internas (envío al profesor, mapa de entregables, Plan B, «la guía pide», «para la sustentación», referencias a AGENTS.md); encabezado con resumen técnico. Limpieza equivalente en docstrings de `src/`. Nuevo `README.md` (contenido, qué no se incluye y por qué, entorno, orden de reproducción). Se arma `entrega_proyecto_final.zip`
- [x] **Cuaderno ejecutado por Felipe (11-nov, 16:06)**: 57/57 celdas, orden consecutivo, 0 errores, 26 cifras clave verificadas contra texto, informe y presentación. `entrega/ProyectoFinal_DL_MotilidadVISEM.zip` generado con `empaquetar_entrega.py --zip`: 18,2 MB, 122 archivos, integridad OK
- [x] **Presentación final (11-nov, versión 3)**: 23 láminas con color por sección, barra de secciones arriba, portada oscura con el frame real del participante 13 y sus 47 trayectorias reales que se dibujan, gráfico de resultados en HTML animado (barras que crecen) y lámina nueva de **análisis de errores** (deriva 10,0 vs 16,4; concentración baja 8,5 vs 12,9; contracción −4,4/−7,7 y +7,0/+14,6; correlación 0,72; participante 7). Se probó y se **quitó el contador animado de cifras**: en la captura quedaba en valores intermedios falsos (0,218 en vez de 0,886); todas las cifras se muestran exactas siempre. Red de seguridad: a los 3 s cualquier animación queda resuelta. PDF de 23 páginas verificado; guion a 23 láminas (14:45; Felipe 7:25, Santiago 7:20); .zip regenerado (18,7 MB; adjuntos 21,1 MB)
- [x] **Presentación visual (v4, 11-nov)**: 21 láminas, una idea por lámina, cifras grandes, imágenes reales (frames, cajas, trayectorias), diagramas de los 3 modelos, flujo con fotos de cada paso; detalles en las notas del orador (N). Cubre todo lo que exige el correo (curvas, hiperparámetros, pérdida, optimizador, cómputo, muestras externas, análisis de errores, ¿responde al problema?). Respaldo de la v3 (23 láminas) en el scratchpad de la sesión. PDF de 21 páginas; guion regenerado desde las notas (14:05; Felipe 6:55, Santiago 7:10)
- [x] **Documento técnico complementario** `reports/documento_tecnico.docx/.pdf` (11 páginas, 5 tablas, 6 figuras): análisis completo tipo artículo; va en el .zip como `documento_tecnico_complementario.pdf`. El informe evaluado sigue siendo `informe_final.pdf` (3 páginas)
- [x] .zip final: 93 archivos, 13,8 MB, sin `reports/figures/` (embebidas en cuaderno y presentación; con ellas los adjuntos llegaban a 25,0 MB, el límite de Gmail). Adjuntos totales: 17,4 MB
- [x] Lámina 2: espermatozoides dibujados con cabeza y flagelo (SMIL): progresiva avanza, no progresiva bate en su sitio dentro de un anillo, inmóvil quieta. `empaquetar_entrega.py` ahora arma el .zip desde los archivos originales (un archivo abierto en la carpeta de revisión ya no lo bloquea). .zip final verificado: 93 archivos, 13,8 MB
- [x] **Guía de estudio (12-nov)**: `reports/guia_estudio.pdf/.docx` (46 páginas, 28 figuras, 10 tablas; incluye «los resultados en palabras simples», «otros datos» y «los intentos, uno por uno»), material interno para la sustentación (no va en el .zip): cada concepto con qué es / por qué lo usamos / qué hicimos, más resultados, errores, limitaciones, historial, 16 preguntas probables, cifras y glosario. Figuras de conceptos y con datos reales en `reports/figures/estudio/` (`src/study_figures.py`); el script del documento (`estudio.js`) está en el scratchpad de la sesión
- [x] **Presentación v5 para la sustentación (18-nov)**: mismas 21 láminas, orden e identidad; más clara y con movimiento útil (rondas de CV animadas en la 5, un mismo espermatozoide recorriendo el flujo en la 7, los 4 modelos con su función en la 8, precisión/recall resaltados sobre la matriz en la 18). Correcciones: la barra de video del resultado (14) ahora es la CNN 3D 9,71/7,00 (antes ResNet 9,00 mientras el texto hablaba de la CNN 3D); MOTA ya no se describe como «% de células perdidas» (13); Ridge «α = 10 elegido por validación» → Ridge elegido frente a bosque/boosting (9); nota de la 6 decía que la normalización usaba estadísticas de entrenamiento (es por clip). Guion regenerado desde las notas (tiempos sin cambiar, 14:05). Respaldo de la v4 en el scratchpad. La versión enviada al profesor (en el .zip) es la v4
- [x] **Apoyo para tabletas (18-nov)**: `reports/apoyo_tableta_v4.pdf` (24 páginas A4 horizontal; una por lámina: lámina grande, mini guion y definiciones de una línea arriba; abajo idea, cifras y pregunta probable a todo el ancho; glosario de 65 términos al final). Material interno, no va en el .zip. Versiones anteriores (`apoyo_tableta.pdf`, `apoyo_tableta_horizontal.pdf`, `apoyo_tableta_v3.pdf`) se pueden borrar
- [x] **Carpeta lista para subir a GitHub (21-nov)**: `python armar_repo_github.py` arma `entrega/subir_a_github/` (151 archivos, 37,4 MB: cuaderno, `src/`, partición, informe, presentación, documento técnico, CSV, figuras y `material_estudio/`) desde una lista explícita y revisa que no haya claves, videos, pesos ni archivos de más de 50 MB. `entrega/INSTRUCCIONES_SUBIDA_GITHUB.md` trae los pasos (clonar el remoto `Elpaipsz/Sperm-motility-DeepLearning` en una carpeta nueva, copiar encima, `git rm 00_proyecto_dl.ipynb`, push sin `--force`). El remoto solo tiene el commit inicial. **Pendiente: el push lo hace Gemini (o Felipe); no se ha subido nada.** Fuera del repo: datos, `models/`, `weights/`, `runs/`, `reports/demo/` (Drive) y `reference-notebooks/` (clave de API ajena). Notebooks `01`–`04` no se suben (están dentro del principal como anexos)
- [ ] **Envío al profesor (11-nov, Felipe)**: `entrega/ProyectoFinal_DL_MotilidadVISEM.zip` + `reports/informe_final.pdf` + `reports/presentacion.pdf` (17,5 MB en total). Después: ensayar con cronómetro (guion 14:05), repartir preguntas, probar en el computador del salón el HTML y un video de `reports/demo/` (67), commit/subida a GitHub y Drive

_Actualizar esta sección al final de cada sesión de trabajo._
