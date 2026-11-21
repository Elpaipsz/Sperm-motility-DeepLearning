"""Video anotado para la demostración: detecciones + trayectorias de los últimos 2 s, coloreadas por velocidad.

Usa las trayectorias ya calculadas (data/processed/tracks/<exp>/<ID>.csv.gz), no vuelve a correr el detector.
Color = velocidad del camino en el último segundo (px/s, escala 0-150): azul/morado = lento, amarillo = rápido.
Uso: python src/demo_video.py <exp_detector> <id1 id2 ...> [--inicio 60] [--dur 20]
Salida: reports/demo/<exp>_participante_<ID>.mp4 (a la velocidad real: 50 fps -> cada 2 frames a 25 fps)
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
cmap = matplotlib.colormaps["plasma"]


def render(exp, pid, start_s=60.0, dur_s=20.0):
    video = next((ROOT / "visem-dataset" / "videos").glob(f"{pid}_*.avi"))
    tr = pd.read_csv(ROOT / "data" / "processed" / "tracks" / exp / f"{pid}.csv.gz")
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS)
    f0, f1 = int(start_s * fps), int((start_s + dur_s) * fps)
    by_frame = {f: g for f, g in tr[(tr.frame >= f0 - int(2 * fps)) & (tr.frame < f1)].groupby("frame")}
    out_dir = ROOT / "reports" / "demo"; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{exp}_participante_{pid}.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (640, 480))
    hist = defaultdict(lambda: deque(maxlen=int(2 * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - int(2 * fps)); n_tracks_shown = []
    for f in range(f0 - int(2 * fps), f1):
        ok, frame = cap.read()
        if not ok:
            break
        g = by_frame.get(f)
        if g is not None:
            for r in g.itertuples():
                hist[r.track_id].append((f, r.cx, r.cy))
        if f < f0 or (f - f0) % 2:
            continue
        alive = set(g.track_id) if g is not None else set()
        for tid in alive:
            pts = np.array(hist[tid]); last = pts[pts[:, 0] > f - fps]
            speed = np.linalg.norm(np.diff(last[:, 1:], axis=0), axis=1).sum() / max((last[-1, 0] - last[0, 0]) / fps, 1e-6) if len(last) > 3 else 0.0
            col = tuple(int(255 * c) for c in cmap(min(speed / 150, 1))[:3][::-1])
            cv2.polylines(frame, [pts[:, 1:].astype(np.int32).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
            r = g[g.track_id == tid].iloc[0]
            cv2.rectangle(frame, (int(r.cx - r.w / 2), int(r.cy - r.h / 2)), (int(r.cx + r.w / 2), int(r.cy + r.h / 2)), col, 1)
        n_tracks_shown.append(len(alive))
        cv2.rectangle(frame, (0, 0), (330, 24), (0, 0, 0), -1)
        cv2.putText(frame, f"participante {pid}  t={f / fps:5.1f}s  detectadas: {len(alive)}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release(); cap.release()
    return out, float(np.mean(n_tracks_shown))


if __name__ == "__main__":
    exp = sys.argv[1]
    ids = [int(a) for a in sys.argv[2:] if a.isdigit()]
    for pid in ids:
        out, m = render(exp, pid)
        print(out, f"| media de células seguidas por frame: {m:.1f} | {out.stat().st_size / 1e6:.1f} MB")
