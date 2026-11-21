"""Calidad del seguimiento (objetivo 1): compara las trayectorias de YOLO+ByteTrack con las de
VISEM-Tracking (labels_ftid) en los primeros 30 s de los participantes con cajas.

Métricas (py-motmetrics, emparejamiento por IoU >= 0,5): MOTA, IDF1, cambios de identidad,
precisión y recall de detección. Solo se evalúan los frames que tienen etiqueta (en el participante 23
faltan 174 frames).
Uso: python src/eval_tracking.py <experimento_detector> <rol: val|test|train> [conf]   -> reports/tracking_eval_<exp>_<rol>_conf<conf>.csv
El umbral de confianza se elige sobre 'val'; 'test' se evalúa una sola vez con el umbral ya elegido.
"""
import sys
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd
from ultralytics import YOLO

from track_videos import exp_imgsz, track_video

ROOT = Path(__file__).resolve().parents[1]
GT_ROOT = ROOT / "VISEM-Tracking" / "extracted" / "VISEM_Tracking_Train_v4" / "Train"
W, H = 640, 480
SECONDS = 30


def iou_distance(a, b, max_iou=0.5):
    """Matriz de distancia 1-IoU entre cajas (x, y, w, h); NaN donde IoU < max_iou (no emparejable).
    Propia porque mm.distances.iou_matrix usa np.asfarray, que NumPy 2 eliminó."""
    out = np.full((len(a), len(b)), np.nan)
    if len(a) == 0 or len(b) == 0:
        return out
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 0:1] + a[:, 2:3], a[:, 1:2] + a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    iou = inter / (a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter)
    return np.where(iou >= max_iou, 1 - iou, np.nan)


def load_gt(pid):
    gt, ids = {}, {}   # ids: id de texto de VISEM-Tracking -> entero (motmetrics exige numéricos)
    for f in (GT_ROOT / str(pid) / "labels_ftid").glob("*_with_ftid.txt"):
        frame = int(f.name.split("_frame_")[1].split("_")[0])
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
        gt[frame] = [(ids.setdefault(r[0], len(ids)), float(r[2]) * W, float(r[3]) * H, float(r[4]) * W, float(r[5]) * H) for r in rows]
    return gt


def evaluate_participant(model, pid, video_path, conf, imgsz=640):
    gt = load_gt(pid)
    pred = track_video(model, video_path, max_seconds=SECONDS, conf=conf, imgsz=imgsz)
    pred_by_frame = {k: g for k, g in pred.groupby("frame")}
    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(gt):
        g = gt[frame]
        p = pred_by_frame.get(frame)
        g_ids = [r[0] for r in g]
        g_box = np.array([[r[1] - r[3] / 2, r[2] - r[4] / 2, r[3], r[4]] for r in g]).reshape(-1, 4)
        if p is None:
            p_ids, p_box = [], np.zeros((0, 4))
        else:
            p_ids = p.track_id.tolist()
            p_box = np.column_stack([p.cx - p.w / 2, p.cy - p.h / 2, p.w, p.h]).astype(float)
        acc.update(g_ids, p_ids, iou_distance(g_box, p_box), frameid=frame)
    s = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches", "num_objects", "precision", "recall"], name=str(pid))
    return s


def main():
    exp, role = sys.argv[1], sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    folds = pd.read_csv(ROOT / "data" / "splits" / "folds.csv")
    ids = folds[folds.detector_role == role].ID.tolist()
    # Se usa el .mp4 de VISEM-Tracking (los 30 s exactos con los que se dibujaron las cajas), no el .avi original
    videos = {pid: GT_ROOT / str(pid) / f"{pid}.mp4" for pid in ids}
    model = YOLO(str(ROOT / "models" / "detector" / exp / "weights" / "best.pt"))
    res = pd.concat([evaluate_participant(model, pid, videos[pid], conf, exp_imgsz(exp)) for pid in ids])
    res.loc["media"] = res.mean()
    out = ROOT / "reports" / f"tracking_eval_{exp}_{role}_conf{conf}.csv"
    res.round(3).to_csv(out)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
