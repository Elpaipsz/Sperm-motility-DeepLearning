"""Línea base por REGLAS (sin aprendizaje de pesos): clasifica cada trayectoria y reporta porcentajes.

Cada trayectoria de >= 1 s (tras corregir la deriva) se asigna a una clase con dos umbrales:
    inmóvil          si  VSL < t_vsl                      (casi no avanza)
    progresiva       si  VSL >= t_vsl  y  LIN >= t_lin    (avanza y lo hace en línea recta)
    no progresiva    en otro caso                          (se mueve pero no avanza en línea recta)
El porcentaje de cada clase es la fracción de trayectorias, igual que hace el cuaderno de referencia `punya-tat`
(allí los umbrales son fijos y arbitrarios; aquí se ajustan con los folds de entrenamiento porque VISEM no trae la
escala µm/píxel y las velocidades están en píxeles/s). Es el análogo de un CASA clásico con 2 parámetros.
Protocolo idéntico al resto: folds de data/splits/folds.csv; el test se evalúa una vez, con los umbrales de los 70.
Uso: python src/rule_baseline.py <exp_detector>  ->  reports/rule_cv_pred_<exp>.csv, rule_test_pred_<exp>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["progressive", "non_progressive", "immotile"]
GRID_VSL = np.array([1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 60])
GRID_LIN = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def percentages(tracks, t_vsl, t_lin):
    still = tracks.vsl < t_vsl
    prog = (~still) & (tracks.lin >= t_lin)
    n = len(tracks)
    return np.array([prog.sum(), n - prog.sum() - still.sum(), still.sum()]) / n * 100


def predict(table, ids, t_vsl, t_lin):
    g = {pid: d for pid, d in table.groupby("ID")}
    return np.array([percentages(g[i], t_vsl, t_lin) for i in ids])


def fit(table, ids, Y):
    best = None
    for tv in GRID_VSL:
        for tl in GRID_LIN:
            m = np.abs(predict(table, ids, tv, tl) - Y).mean()
            if best is None or m < best[0]:
                best = (m, tv, tl)
    return best


def main():
    exp = sys.argv[1]
    table = pd.read_csv(ROOT / "data" / "processed" / f"track_table_{exp}.csv.gz")
    folds = pd.read_csv(ROOT / "data" / "splits" / "folds.csv").set_index("ID")
    tv, te = folds[folds.split == "trainval"], folds[folds.split == "test"]
    cv_rows, chosen = [], []
    for k in sorted(tv.fold.unique()):
        tr, va = tv[tv.fold != k], tv[tv.fold == k]
        m, t_vsl, t_lin = fit(table, tr.index.tolist(), tr[TARGETS].values)
        P = predict(table, va.index.tolist(), t_vsl, t_lin)
        chosen.append(dict(fold=k, t_vsl=t_vsl, t_lin=t_lin, mae_train=m))
        cv_rows.append(pd.DataFrame(P, index=va.index, columns=[f"pred_{t}" for t in TARGETS]).assign(fold=k))
    cv = pd.concat(cv_rows)
    m, t_vsl, t_lin = fit(table, tv.index.tolist(), tv[TARGETS].values)
    P = predict(table, te.index.tolist(), t_vsl, t_lin)
    tep = pd.DataFrame(P, index=te.index, columns=[f"pred_{t}" for t in TARGETS])
    cv.rename_axis("ID").reset_index().to_csv(ROOT / "reports" / f"rule_cv_pred_{exp}.csv", index=False)
    tep.rename_axis("ID").reset_index().to_csv(ROOT / "reports" / f"rule_test_pred_{exp}.csv", index=False)
    pd.DataFrame(chosen + [dict(fold="todos(70)", t_vsl=t_vsl, t_lin=t_lin, mae_train=m)]).to_csv(ROOT / "reports" / f"rule_thresholds_{exp}.csv", index=False)
    mae_cv = np.mean([np.abs(cv[cv.fold == k][[f"pred_{t}" for t in TARGETS]].values - tv.loc[cv[cv.fold == k].index, TARGETS].values).mean() for k in sorted(tv.fold.unique())])
    print("umbrales por fold:\n", pd.DataFrame(chosen).to_string(index=False))
    print(f"umbrales con los 70: t_vsl={t_vsl} px/s, t_lin={t_lin}")
    print(f"MAE CV {mae_cv:.2f} | MAE test {np.abs(P - te[TARGETS].values).mean():.2f}")


if __name__ == "__main__":
    main()
