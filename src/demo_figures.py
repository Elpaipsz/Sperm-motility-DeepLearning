"""Figuras de demostración: se ven los espermatozoides detectados (cajas reales contra cajas predichas).

  1. demo_evisan_imagenes.png : una imagen de validación (200x) y una de test (600x) de EVISAN, cajas reales y del detector A.
  2. demo_visem_test.png      : un frame de cada participante de TEST de VISEM con anotación (13, 14, 35, 60):
                                real | detector A (solo EVISAN) | B | D.
  3. demo_evisan_videos.png   : los 2 videos de prueba de EVISAN (independientes de las imágenes de entrenamiento),
                                evaluación externa del detector A. No traen anotaciones: se muestran frames con
                                detecciones y el conteo por frame; no se calcula mAP.
Uso: python src/demo_figures.py   ->  reports/figures/demo_*.png, reports/evisan_test_videos.csv
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
CONF = 0.4   # el mismo umbral que se usa en el seguimiento y en la estimación de motilidad
DET = {"A": ("A_evisan_only", 640), "B": ("B_evisan_to_visem", 640), "D": ("D_evisan_to_visem_960", 960)}
EXTRA = dict(a.split("=") for a in sys.argv[1:])   # p. ej. E=E2_evisan_to_visem_s960:960


def load(exp):
    return YOLO(str(ROOT / "models" / "detector" / exp / "weights" / "best.pt"))


def gt_boxes(label_file, w, h):
    rows = [l.split() for l in Path(label_file).read_text().splitlines() if l.strip()]
    return [((float(x) - float(bw) / 2) * w, (float(y) - float(bh) / 2) * h, float(bw) * w, float(bh) * h) for _, x, y, bw, bh in rows]


def pred_boxes(model, img, imgsz):
    r = model.predict(img, conf=CONF, imgsz=imgsz, verbose=False, device=0)[0]
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy()]


def draw(ax, img, boxes, color, title, lw=1.0):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    for x, y, w, h in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=lw))
    ax.set_title(title, fontsize=9); ax.axis("off")


def fig_evisan_images(models):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6))
    for r, (split, name, mag) in enumerate([("val", "2500", "200x"), ("test", "5500", "600x")]):
        img = cv2.imread(str(DATA / "evisan_yolo" / split / "images" / f"{name}.jpg"), 0); h, w = img.shape
        gt = gt_boxes(DATA / "evisan_yolo" / split / "labels" / f"{name}.txt", w, h)
        pr = pred_boxes(models["A"], img, DET["A"][1])
        draw(ax[r, 0], img, gt, "lime", f"EVISAN {split} · {mag} · real ({len(gt)} espermatozoides)")
        draw(ax[r, 1], img, pr, "red", f"detector A (solo EVISAN) · {len(pr)} detectados")
    plt.tight_layout(); plt.savefig(FIG / "demo_evisan_imagenes.png", dpi=110); plt.close()


def fig_visem_test(models):
    pids = [13, 14, 35, 60]; names = ["A"] + ["B", "D"] + [k for k in EXTRA]
    fig, ax = plt.subplots(len(pids), 1 + len(names), figsize=(3.7 * (1 + len(names)), 2.9 * len(pids)))
    rows = []
    for r, pid in enumerate(pids):
        name = f"{pid}_frame_100"
        img = cv2.imread(str(DATA / "visem_yolo" / "test" / "images" / f"{name}.jpg"), 0); h, w = img.shape
        gt = gt_boxes(DATA / "visem_yolo" / "test" / "labels" / f"{name}.txt", w, h)
        draw(ax[r, 0], img, gt, "lime", f"participante {pid} · real ({len(gt)})", lw=0.9)
        for c, n in enumerate(names, 1):
            pr = pred_boxes(models[n], img, DET[n][1] if n in DET else int(EXTRA[n].split(":")[1]))
            draw(ax[r, c], img, pr, "red", f"detector {n} ({len(pr)} detectados)", lw=0.9)
            rows.append(dict(participante=pid, detector=n, reales=len(gt), detectados=len(pr)))
    plt.tight_layout(); plt.savefig(FIG / "demo_visem_test.png", dpi=105); plt.close()
    return pd.DataFrame(rows)


def fig_evisan_videos(models):
    fig, ax = plt.subplots(2, 4, figsize=(16, 7)); stats = []
    for r, vid in enumerate(["test_200x", "test_600x"]):
        cap = cv2.VideoCapture(str(ROOT / "EVISAN" / "video" / f"{vid}.wmv")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        counts, idx = [], list(range(0, n, 10))
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (1200, int(round(g.shape[0] * 1200 / g.shape[1]))))   # al tamaño de las imágenes de entrenamiento
            pr = pred_boxes(models["A"], g, DET["A"][1]); counts.append(len(pr))
            if i in (idx[len(idx) // 5], idx[len(idx) // 2], idx[4 * len(idx) // 5]):
                draw(ax[r, [idx[len(idx) // 5], idx[len(idx) // 2], idx[4 * len(idx) // 5]].index(i)], g, pr, "red", f"{vid} · frame {i} · {len(pr)} detectados")
        ax[r, 3].plot(np.array(idx[:len(counts)]) / 18.25, counts); ax[r, 3].set_xlabel("tiempo (s)"); ax[r, 3].set_ylabel("detecciones por frame")
        ax[r, 3].set_title(f"{vid}: conteo por frame (cada 10 frames)", fontsize=9)
        stats.append(dict(video=vid, frames_muestreados=len(counts), media=np.mean(counts), desv=np.std(counts), minimo=min(counts), maximo=max(counts),
                          resolucion=f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"))
    plt.tight_layout(); plt.savefig(FIG / "demo_evisan_videos.png", dpi=95); plt.close()
    return pd.DataFrame(stats)


def main():
    models = {k: load(v[0]) for k, v in DET.items()}
    models.update({k: load(v.split(":")[0]) for k, v in EXTRA.items()})
    fig_evisan_images(models)
    vt = fig_visem_test(models); vt.to_csv(ROOT / "reports" / "demo_conteos_visem.csv", index=False); print(vt.to_string(index=False))
    s = fig_evisan_videos(models); s.round(1).to_csv(ROOT / "reports" / "evisan_test_videos.csv", index=False); print(s.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
