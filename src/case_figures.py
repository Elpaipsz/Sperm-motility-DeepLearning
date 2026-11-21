"""Figuras de casos: un frame real del video con las trayectorias detectadas de los
últimos 3 s (coloreadas por velocidad) y, al lado, motilidad real frente a la estimada por cada método.

Uso: python src/case_figures.py <experimento_detector> <id1 id2 ...>   ->  reports/figures/casos/caso_<ID>.png
Las predicciones son las de validación cruzada / test (el modelo que predice a cada participante no lo vio).
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_utils import (ROOT, TARGETS, load_cnn_predictions, load_folds, load_regression_predictions)

T_FRAME_S, TRAIL_S = 60.0, 3.0
LABELS = ["progresiva", "no progresiva", "inmóvil"]


def read_frame(video, t_s):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = int(round(t_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ok else None), fps, idx


def case_figure(pid, exp, folds, preds, out_dir):
    video = next((ROOT / "visem-dataset" / "videos").glob(f"{pid}_*.avi"))
    frame, fps, f0 = read_frame(video, T_FRAME_S)
    tracks = pd.read_csv(ROOT / "data" / "processed" / "tracks" / exp / f"{pid}.csv.gz")
    win = tracks[(tracks.frame > f0 - TRAIL_S * fps) & (tracks.frame <= f0)]
    alive = set(win[win.frame == f0].track_id)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    a0.imshow(frame, cmap="gray"); a0.axis("off")
    speeds = {}
    for tid, g in win[win.track_id.isin(alive)].groupby("track_id"):
        if len(g) < 5:
            continue
        xy = g[["cx", "cy"]].values
        speeds[tid] = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / ((g.frame.max() - g.frame.min()) / fps)
        g_xy = xy
        a0.plot(g_xy[:, 0], g_xy[:, 1], lw=1.4, color=plt.cm.plasma(min(speeds[tid] / 150, 1)))
    row = folds.loc[pid]
    note = f" · nota: {row.quality_note}" if isinstance(row.quality_note, str) else ""
    a0.set_title(f"Participante {pid} · t = {T_FRAME_S:.0f} s · {len(speeds)} trayectorias visibles (últimos {TRAIL_S:.0f} s){note}", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 150)); cb = plt.colorbar(sm, ax=a0, fraction=0.035, pad=0.01)
    cb.set_label("velocidad del camino (px/s)", fontsize=8)

    real = row[TARGETS].values.astype(float)
    x = np.arange(3); w = 0.26
    a1.bar(x - w, real, w, label="real (laboratorio)", color="tab:gray")
    for k, (name, p) in enumerate(preds.items()):
        if pid in p.index:
            a1.bar(x + k * w, p.loc[pid].values, w, label=name)
    a1.set_xticks(x); a1.set_xticklabels(LABELS); a1.set_ylabel("%"); a1.set_ylim(0, 100); a1.legend(fontsize=8)
    a1.set_title(f"Motilidad · evaluador {row.evaluator}", fontsize=9)
    plt.tight_layout()
    out = out_dir / f"caso_{pid}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    exp, ids = sys.argv[1], [int(i) for i in sys.argv[2:]]
    folds = load_folds()
    rep = ROOT / "reports"
    ridge_cv, ridge_te = load_regression_predictions(exp)
    cnn_cv, cnn_te = load_cnn_predictions([rep / "baseline_predictions.csv", rep / "baseline_predictions_s7.csv", rep / "baseline_predictions_s123.csv"])
    preds = {"Trayectorias + Ridge": pd.concat([ridge_cv, ridge_te]) , "CNN 3D (video directo)": pd.concat([cnn_cv, cnn_te])}
    preds = {k: v[["pred_progressive", "pred_non_progressive", "pred_immotile"]].set_axis(range(3), axis=1) for k, v in preds.items()}
    out_dir = rep / "figures" / "casos"; out_dir.mkdir(parents=True, exist_ok=True)
    for pid in ids:
        print(case_figure(pid, exp, folds, preds, out_dir))


if __name__ == "__main__":
    main()
