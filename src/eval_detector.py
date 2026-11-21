"""Evalúa los detectores (A, B, C, D por defecto; o los pasados como argumento) en EVISAN-test (secuencia 6) y VISEM-Tracking (val y test).

El mAP sobre VISEM-test (4 participantes que ningún detector vio) es la medida de transferencia/generalización.
Guarda reports/detector_eval.csv.
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from track_videos import exp_imgsz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "detector"

SETS = {
    "EVISAN_test": (DATA / "evisan_yolo" / "data.yaml", "test"),
    "VISEM_val": (DATA / "visem_yolo" / "data.yaml", "val"),
    "VISEM_test": (DATA / "visem_yolo" / "data.yaml", "test"),
}


def main():
    rows = []
    exps = sys.argv[1:] or ["A_evisan_only", "B_evisan_to_visem", "C_visem_only", "D_evisan_to_visem_960"]
    for exp in exps:
        weights = MODELS / exp / "weights" / "best.pt"
        if not weights.exists():
            print("sin pesos:", exp)
            continue
        model = YOLO(str(weights))
        for name, (yaml, split) in SETS.items():
            m = model.val(data=str(yaml), split=split, imgsz=exp_imgsz(exp), batch=16, workers=0, device=0,
                          plots=False, verbose=False).box
            rows.append(dict(experiment=exp, eval_set=name, precision=m.mp, recall=m.mr, mAP50=m.map50, mAP50_95=m.map))
    out = pd.DataFrame(rows).round(3)
    path = ROOT / "reports" / "detector_eval.csv"
    if sys.argv[1:] and path.exists():   # añade o reemplaza solo los experimentos pedidos
        old = pd.read_csv(path); out = pd.concat([old[~old.experiment.isin(exps)], out], ignore_index=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
