"""Extrae UNA SOLA VEZ los videos de VISEM a disco: se decodifica una vez y se trabaja desde los archivos extraídos.

Por participante: 4 ventanas de 30 s de tiempo REAL (0-120 s), cada una como array uint8 (750, 240, 320)
en escala de grises a 25 fps efectivos (índice de frame = round(t * fps_real)).
- Se usa tiempo real y no número de frames porque el fps varía entre videos (37,3 a 50,2): 3 videos
  rondan 37-44 fps. Todos duran >= 120 s.
- 25 fps y 320x240 bastan para la línea base de video; el seguimiento (velocidad) se hará sobre el
  video original a su fps real, no sobre estos arrays.
- Salida: data/processed/visem_gray/{ID}_w{0..3}.npy y data/processed/visem_gray/manifest.csv.
Es reanudable: si el .npy ya existe se omite.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "visem-dataset" / "videos"
OUT = ROOT / "data" / "processed" / "visem_gray"
WINDOW_S, N_WINDOWS, OUT_FPS, SIZE = 30, 4, 25, (320, 240)   # SIZE = (ancho, alto)
N_OUT = WINDOW_S * OUT_FPS


def extract_video(path, pid):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = {}   # índice de frame -> (ventana, posición)
    for w in range(N_WINDOWS):
        for k in range(N_OUT):
            idx = min(round((w * WINDOW_S + k / OUT_FPS) * fps), n_frames - 1)
            wanted.setdefault(idx, []).append((w, k))
    arrays = {w: np.zeros((N_OUT, SIZE[1], SIZE[0]), np.uint8) for w in range(N_WINDOWS)}
    last = max(wanted)
    for i in range(last + 1):
        if i not in wanted:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), SIZE, interpolation=cv2.INTER_AREA)
        for w, k in wanted[i]:
            arrays[w][k] = small
    cap.release()
    for w, arr in arrays.items():
        np.save(OUT / f"{pid}_w{w}.npy", arr)
    return dict(ID=pid, fps=round(fps, 3), n_frames=n_frames, duration_s=round(n_frames / fps, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(VIDEOS.glob("*.avi"), key=lambda p: int(p.name.split("_")[0])):
        pid = int(path.name.split("_")[0])
        if all((OUT / f"{pid}_w{w}.npy").exists() for w in range(N_WINDOWS)):
            continue
        rows.append(extract_video(path, pid))
        print("ok", pid, rows[-1], flush=True)
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(OUT / "manifest.csv") if (OUT / "manifest.csv").exists() else pd.DataFrame()
        pd.concat([old, new]).drop_duplicates("ID").sort_values("ID").to_csv(OUT / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
