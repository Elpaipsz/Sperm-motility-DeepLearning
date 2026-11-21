"""Entrena y guarda el SISTEMA FINAL con los 70 participantes de entrenamiento (trainval), sin tocar el test.

Sistema final (sección 5.8 del cuaderno): detector B -> ByteTrack -> trayectorias -> promedio de
  (a) Ridge sobre 24 descriptores por participante y (b) red de conjuntos sobre las trayectorias (3 semillas).
Además guarda lo que necesita la predicción de un video nuevo (src/predict_video.py):
  - semiancho del intervalo conformal al 80 % y 90 % por variable, calibrado con los residuos de la CV (70)
    de este mismo sistema (Anexo D);
  - umbral de "pocas trayectorias": el tercil inferior de trayectorias largas en los 70; en ese tercio el
    error de CV es mayor (sección 5.5.1), así que la predicción se marca como menos confiable.
Uso: python src/final_system.py   ->  models/final/{ridge.joblib, setnet_s*.pt, config.json}
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import track_set_net as T   # noqa: E402
from results_utils import PRED, TARGETS, load_cnn_predictions, load_folds, load_regression_predictions   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = "B_evisan_to_visem"
SEEDS = (42, 7, 123)
OUT = ROOT / "models" / "final"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = load_folds(); tv = F[F.split == "trainval"]
    feats = pd.read_csv(ROOT / "data" / "processed" / f"features_{EXP}.csv").set_index("ID")
    cols = list(feats.columns)
    # (a) Ridge: un modelo por variable, igual que src/motility_regression.py (alpha = 10, estandarizado)
    ridge = [make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(feats.loc[tv.index, cols].values, tv[t].values) for t in TARGETS]
    joblib.dump(dict(models=ridge, cols=cols), OUT / "ridge.joblib")
    # (b) red de conjuntos: misma configuración que el cuaderno (32 unidades, dropout 0,2, 300 épocas), con los 70
    table = T.build_track_table(EXP)
    tracks = {pid: dict(X=T.track_matrix(g), t0=g.t0.values) for pid, g in table.groupby("ID")}
    Y = {i: F.loc[i, TARGETS].values.astype(float) for i in F.index}
    for s in SEEDS:
        net, mu, sd, _ = T.train_fold(tv.index.tolist(), tracks, Y, s, 32, 0.2)
        torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, hidden=32, p_drop=0.2), OUT / f"setnet_s{s}.pt")
        print("red de conjuntos entrenada, semilla", s, flush=True)
    # intervalo conformal con los residuos de CV del sistema final (predicciones fuera de muestra ya guardadas)
    r_cv = load_regression_predictions(EXP)[0]
    s_cv = load_cnn_predictions([ROOT / "reports" / f"baseline_predictions_trackset_s{k}.csv" for k in SEEDS])[0]
    fin = (r_cv.loc[tv.index] + s_cv.loc[tv.index]) / 2
    res = np.abs(fin.values - tv[TARGETS].values); n = len(res)
    q = {f"{int(lvl * 100)}": [float(np.quantile(res[:, j], min(1, np.ceil((n + 1) * lvl) / n))) for j in range(3)] for lvl in (0.8, 0.9)}
    cfg = dict(detector=EXP, conf=0.4, imgsz=640, seconds=120, targets=TARGETS, seeds=list(SEEDS),
               conformal_halfwidth=q, low_tracks_threshold=float(np.quantile(feats.loc[tv.index, "n_long_tracks"], 1 / 3)),
               cv_mae_final=float(np.abs(fin.values - tv[TARGETS].values).mean()), trained_on="70 participantes trainval (folds.csv)")
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
