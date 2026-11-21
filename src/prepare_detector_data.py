"""Prepara los datasets YOLO (una sola clase: 'sperm') en data/processed/.

- EVISAN: XML Pascal VOC -> YOLO; split por SECUENCIA de 1.000 imágenes
  (train: 1, 2, 4, 5 | val: 3 (200X) | test: 6 (600X)).
- VISEM-Tracking: split por PARTICIPANTE según data/splits/folds.csv (columna detector_role);
  se toma 1 de cada STRIDE frames porque frames consecutivos son casi idénticos.
- Las clases se fusionan en una sola: en EVISAN black/white no están documentadas; en VISEM-Tracking
  normal/clúster/pinhead se tratan como "objeto a contar" para que los mAP sean comparables.
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVISAN = ROOT / "EVISAN"
TRACKING = ROOT / "VISEM-Tracking" / "extracted" / "VISEM_Tracking_Train_v4" / "Train"
OUT = ROOT / "data" / "processed"
STRIDE = 5

EVISAN_SPLIT = {"train": (1, 2, 4, 5), "val": (3,), "test": (6,)}   # índice de secuencia (1..6)


def voc_to_yolo_lines(xml_path):
    root = ET.parse(xml_path).getroot()
    w, h = float(root.find("size/width").text), float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        x0, y0, x1, y1 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def prepare_evisan():
    base = OUT / "evisan_yolo"
    counts = {}
    for split, seqs in EVISAN_SPLIT.items():
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
        n = 0
        for seq in seqs:
            for i in range((seq - 1) * 1000 + 1, seq * 1000 + 1):
                shutil.copy2(EVISAN / "images" / f"{i}.jpg", base / split / "images" / f"{i}.jpg")
                (base / split / "labels" / f"{i}.txt").write_text("\n".join(voc_to_yolo_lines(EVISAN / "xml_file" / f"{i}.xml")))
                n += 1
        counts[split] = n
    (base / "data.yaml").write_text(
        f"path: {base.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: sperm\n")
    return counts


def prepare_tracking():
    folds = pd.read_csv(ROOT / "data" / "splits" / "folds.csv")
    base = OUT / "visem_yolo"
    counts = {}
    for _, row in folds[folds.has_tracking].iterrows():
        split = {"train": "train", "val": "val", "test": "test"}[row.detector_role]
        pid = int(row.ID)
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in sorted((TRACKING / str(pid) / "images").glob("*.jpg"),
                          key=lambda p: int(p.stem.rsplit("_", 1)[1]))[::STRIDE]:
            lab = TRACKING / str(pid) / "labels" / f"{img.stem}.txt"
            if not lab.exists():   # participante 23: 174 frames sin etiqueta ni fila en el CSV de conteos; no se asumen vacíos
                continue
            lines = [f"0 {' '.join(l.split()[1:5])}" for l in lab.read_text().splitlines() if l.strip()]
            shutil.copy2(img, base / split / "images" / img.name)
            (base / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines))
            counts[split] = counts.get(split, 0) + 1
    (base / "data.yaml").write_text(
        f"path: {base.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: sperm\n")
    return counts


if __name__ == "__main__":
    print("EVISAN (imágenes por split):", prepare_evisan())
    print("VISEM-Tracking (frames por split, 1 de cada %d):" % STRIDE, prepare_tracking())
