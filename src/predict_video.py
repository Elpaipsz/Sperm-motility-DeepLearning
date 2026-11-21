"""Sistema final de punta a punta: un video de microscopía -> motilidad estimada con intervalo y aviso.

Pasos: detector B (YOLO11n) + ByteTrack sobre los primeros 120 s -> corrección de deriva -> (a) 24 descriptores
-> Ridge y (b) trayectorias individuales -> red de conjuntos (3 semillas) -> promedio de (a) y (b), normalizado a 100.
Intervalo: conformal por variable (80 % o 90 %), calibrado con la CV de los 70 participantes de entrenamiento.
Aviso: si hay pocas trayectorias (tercil inferior del entrenamiento) el error de CV fue mayor (9,5 contra 5,7 pp).

Válido SOLO para videos como los de VISEM (microscopio Olympus CX31, 400x, 640x480, ~50 fps). En otro equipo la
escala y el aspecto cambian y nada de esto se ha validado. No diagnostica fertilidad: estima porcentajes de motilidad.
Uso: python src/predict_video.py <video.avi> [--nivel 90]
Requiere haber corrido antes: python src/final_system.py
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import track_set_net as T   # noqa: E402
from motion_features import features_from_tracks, remove_drift, track_metrics   # noqa: E402
from track_videos import track_video   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "models" / "final"
LABELS = {"progressive": "progresiva", "non_progressive": "no progresiva", "immotile": "inmóvil"}


def track_table(df, fps):
    """Una fila por trayectoria de >= 1 s (misma definición que track_set_net.build_track_table)."""
    df = remove_drift(df); rows = []
    for _, tr in df.groupby("track_id"):
        tr = tr.sort_values("frame"); m = track_metrics(tr, fps)
        if m is not None:
            rows.append(dict(t0=tr.t_s.iloc[0], dur=tr.t_s.iloc[-1] - tr.t_s.iloc[0], **m))
    return pd.DataFrame(rows)


def predict(video, level="90"):
    cfg = json.loads((FINAL / "config.json").read_text(encoding="utf-8"))
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    det = YOLO(str(ROOT / "models" / "detector" / cfg["detector"] / "weights" / "best.pt"))
    tracks = track_video(det, video, max_seconds=cfg["seconds"], conf=cfg["conf"], imgsz=cfg["imgsz"])
    if tracks.empty:
        raise SystemExit("No se detectó ningún espermatozoide: el video no se parece a los de entrenamiento.")
    # (a) Ridge sobre los descriptores del participante
    rd = joblib.load(FINAL / "ridge.joblib")
    f = pd.Series(features_from_tracks(tracks, fps)).reindex(rd["cols"]).fillna(0.0)
    p_ridge = np.clip([m.predict(f.values[None, :])[0] for m in rd["models"]], 0, None); p_ridge = p_ridge / p_ridge.sum() * 100
    # (b) red de conjuntos sobre las trayectorias individuales
    tt = track_table(tracks, fps)
    trk = {0: dict(X=T.track_matrix(tt), t0=tt.t0.values)}
    p_set = []
    for s in cfg["seeds"]:
        ck = torch.load(FINAL / f"setnet_s{s}.pt", weights_only=False)
        net = T.SetNet(hidden=ck["hidden"], p_drop=ck["p_drop"]); net.load_state_dict(ck["state"])
        p_set.append(T.predict(net, [0], trk, ck["mu"], ck["sd"])[0])
    p_set = np.mean(p_set, axis=0)
    p = (p_ridge + p_set) / 2
    hw = np.array(cfg["conformal_halfwidth"][level])
    n_long = int(f["n_long_tracks"])
    return dict(video=str(video), prediccion={LABELS[t]: round(float(v), 1) for t, v in zip(cfg["targets"], p)},
                intervalo={LABELS[t]: [round(float(max(0, v - h)), 1), round(float(min(100, v + h)), 1)] for t, v, h in zip(cfg["targets"], p, hw)},
                nivel_intervalo=f"{level} %", ridge={LABELS[t]: round(float(v), 1) for t, v in zip(cfg["targets"], p_ridge)},
                red_de_conjuntos={LABELS[t]: round(float(v), 1) for t, v in zip(cfg["targets"], p_set)},
                trayectorias_largas=n_long, detecciones_por_frame=round(float(f["det_per_frame"]), 1),
                aviso=("POCAS TRAYECTORIAS: en este caso el error de validación fue mayor (≈ 9,5 pp de MAE frente a ≈ 5,7)."
                       if n_long < cfg["low_tracks_threshold"] else None)), tracks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--nivel", default="90", choices=["80", "90"])
    a = ap.parse_args()
    out, tracks = predict(Path(a.video), a.nivel)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
