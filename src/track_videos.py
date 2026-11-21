"""Detecta y sigue espermatozoides (YOLO + ByteTrack) en los primeros 120 s de cada video de VISEM.

Corre sobre el video ORIGINAL a su fps real (no sobre los arrays reducidos), porque la velocidad
depende del fps y del tamaño de píxel.
Uso: python src/track_videos.py <experimento_detector> [id1 id2 ...]
Salida: data/processed/tracks/<experimento>/<ID>.csv.gz con columnas
        frame, t_s, track_id, cx, cy, w, h, conf   (coordenadas en píxeles de 640x480)
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "visem-dataset" / "videos"
MAX_SECONDS = int(__import__("os").environ.get("TRACK_SECONDS", 120))   # 30 para comparaciones rápidas (Anexo C: 30 s bastan)
CONF = 0.10   # umbral bajo: ByteTrack recupera detecciones débiles en una segunda asociación


def exp_imgsz(exp):
    """Resolución con la que se entrenó el detector (D se entrenó a 960; el resto a 640)."""
    return 960 if "960" in exp else 640


def track_video(model, path, max_seconds=MAX_SECONDS, conf=CONF, imgsz=640):
    import cv2
    fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS)
    rows = []
    stream = model.track(source=str(path), stream=True, tracker="bytetrack.yaml", conf=conf, imgsz=imgsz,
                         device=0, verbose=False, persist=False)
    for frame_idx, r in enumerate(stream):
        if frame_idx / fps > max_seconds:
            break
        b = r.boxes
        if b is None or b.id is None:
            continue
        xywh, ids, conf = b.xywh.cpu().numpy(), b.id.cpu().numpy().astype(int), b.conf.cpu().numpy()
        for (cx, cy, w, h), tid, c in zip(xywh, ids, conf):
            rows.append((frame_idx, frame_idx / fps, tid, cx, cy, w, h, c))
    return pd.DataFrame(rows, columns=["frame", "t_s", "track_id", "cx", "cy", "w", "h", "conf"])


def main():
    exp = sys.argv[1]
    only = {int(x) for x in sys.argv[2:]}
    model = YOLO(str(ROOT / "models" / "detector" / exp / "weights" / "best.pt"))
    out = ROOT / "data" / "processed" / "tracks" / (exp + ("" if MAX_SECONDS == 120 else f"_{MAX_SECONDS}s"))
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(VIDEOS.glob("*.avi"), key=lambda p: int(p.name.split("_")[0])):
        pid = int(path.name.split("_")[0])
        if (only and pid not in only) or (out / f"{pid}.csv.gz").exists():
            continue
        df = track_video(model, path, max_seconds=MAX_SECONDS, conf=float(__import__("os").environ.get("TRACK_CONF", CONF)), imgsz=exp_imgsz(exp))
        df.to_csv(out / f"{pid}.csv.gz", index=False)
        print("ok", pid, len(df), "detecciones,", df.track_id.nunique(), "trayectorias", flush=True)


if __name__ == "__main__":
    main()
