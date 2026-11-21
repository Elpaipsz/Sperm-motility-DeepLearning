"""Entrena los detectores YOLO (una clase: sperm) y guarda todo en models/detector/<nombre>/.

Experimentos (misma arquitectura yolo11n, imgsz 640, semilla 42, AMP activado por defecto):
  A  evisan_only      : COCO -> EVISAN (train: secuencias 1,2,4,5; val: secuencia 3)
  B  evisan_to_visem  : A -> afinado con VISEM-Tracking (participantes de train; val: fold 0)
  C  visem_only       : COCO -> VISEM-Tracking (control: ¿sirve de algo pasar por EVISAN?)

imgsz=640: a 1200 px de ancho las cajas de EVISAN miden ~40 px; reescaladas a 640 quedan en ~21 px,
casi igual que las de VISEM-Tracking (~23 px a resolución nativa). Así el detector ve objetos del
mismo tamaño en ambos dominios.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "detector"
SEED = 42
COCO = ROOT / "models" / "pretrained" / "yolo11n.pt"   # pesos preentrenados en COCO (punto de partida)

EXPERIMENTS = {
    "A_evisan_only": dict(weights=COCO, data=DATA / "evisan_yolo" / "data.yaml", epochs=50),
    "B_evisan_to_visem": dict(weights=OUT / "A_evisan_only" / "weights" / "best.pt",
                              data=DATA / "visem_yolo" / "data.yaml", epochs=30),
    "C_visem_only": dict(weights=COCO, data=DATA / "visem_yolo" / "data.yaml", epochs=50),
    # D: A -> VISEM-Tracking con más resolución (objetos de ~23 px) y aumentos más fuertes de color y
    #    volteo vertical (en microscopía no hay "arriba"); motivado por el fallo en el participante 19
    #    (residuo y contraste distintos, 12 participantes de entrenamiento no cubren esa variación).
    "D_evisan_to_visem_960": dict(weights=OUT / "A_evisan_only" / "weights" / "best.pt",
                                  data=DATA / "visem_yolo" / "data.yaml", epochs=30, imgsz=960, batch=8,
                                  extra=dict(flipud=0.5, hsv_s=0.6, hsv_v=0.5, lr0=0.003)),
    # E: mismo recorrido que B/D con un modelo mayor (yolo11s, 9,4 M de parámetros contra 2,6 M) y parada temprana
    #    más corta: en B/C/D la mejor época de validación fue la 1 o la 2 (sobreajuste), así que se baja el lr y se
    #    corta con paciencia 5. Etapa 1: COCO -> EVISAN (640 px). Etapa 2: -> VISEM-Tracking a 960 px.
    "E1_evisan_s": dict(weights=ROOT / "models" / "pretrained" / "yolo11s.pt", data=DATA / "evisan_yolo" / "data.yaml", epochs=50),
    "E2_evisan_to_visem_s960": dict(weights=OUT / "E1_evisan_s" / "weights" / "best.pt",
                                    data=DATA / "visem_yolo" / "data.yaml", epochs=30, imgsz=960, batch=8, patience=5,
                                    extra=dict(flipud=0.5, hsv_s=0.6, hsv_v=0.5, lr0=0.001)),
}


def run(name):
    cfg = EXPERIMENTS[name]
    model = YOLO(str(cfg["weights"]))
    model.train(data=str(cfg["data"]), epochs=cfg["epochs"], imgsz=cfg.get("imgsz", 640), batch=cfg.get("batch", 16),
                workers=0, device=0, seed=SEED, patience=cfg.get("patience", 15), project=str(OUT), name=name, exist_ok=True,
                plots=True, **cfg.get("extra", {}))


if __name__ == "__main__":
    for exp in (sys.argv[1:] or ["A_evisan_only", "B_evisan_to_visem", "C_visem_only"]):
        run(exp)
