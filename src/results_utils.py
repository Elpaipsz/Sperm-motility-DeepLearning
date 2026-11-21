"""Utilidades para comparar modelos de motilidad con el MISMO protocolo (las usan el cuaderno y sus anexos).

Protocolo: la CV se evalúa sobre los 70 participantes de 'trainval' con los folds de data/splits/folds.csv;
el test son los 15 participantes reservados. Las predicciones de test de los modelos de fold se promedian.
Todas las cifras son MAE/RMSE en puntos porcentuales (pp), promediados sobre las tres variables.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["progressive", "non_progressive", "immotile"]
PRED = ["pred_" + t for t in TARGETS]


def load_folds():
    return pd.read_csv(ROOT / "data" / "splits" / "folds.csv").set_index("ID")


def load_cnn_predictions(paths):
    """Lee uno o varios CSV de src/baseline_video.py (o variantes) y promedia entre semillas.
    Devuelve (cv, test): DataFrames indexados por ID con columnas pred_*."""
    cvs, tes = [], []
    for p in paths:
        d = pd.read_csv(p)
        cvs.append(d[d.split == "cv"].groupby("ID")[PRED].mean())
        tes.append(d[d.split == "test"].groupby("ID")[PRED].mean())
    return sum(cvs) / len(cvs), sum(tes) / len(tes)


def load_regression_predictions(exp):
    cv = pd.read_csv(ROOT / "reports" / f"regression_cv_pred_{exp}.csv").set_index("ID")[PRED]
    te = pd.read_csv(ROOT / "reports" / f"regression_test_pred_{exp}.csv").set_index("ID")[PRED]
    return cv, te


def score(cv, te, folds=None):
    """MAE/RMSE de CV (media ± desviación entre folds) y de test para un par de predicciones."""
    folds = load_folds() if folds is None else folds
    y_cv, y_te = folds.loc[cv.index, TARGETS].values, folds.loc[te.index, TARGETS].values
    fold_of = folds.loc[cv.index, "fold"]
    per_fold = [np.abs(cv.loc[fold_of[fold_of == k].index].values - folds.loc[fold_of[fold_of == k].index, TARGETS].values).mean()
                for k in sorted(fold_of.unique())]
    return dict(cv_mae=np.mean(per_fold), cv_mae_std=np.std(per_fold),
                cv_rmse=np.sqrt(((cv.values - y_cv) ** 2).mean()),
                test_mae=np.abs(te.values - y_te).mean(), test_rmse=np.sqrt(((te.values - y_te) ** 2).mean()),
                cv_mae_prog=np.abs(cv.values[:, 0] - y_cv[:, 0]).mean(), test_mae_prog=np.abs(te.values[:, 0] - y_te[:, 0]).mean())


def trivial_predictions(folds=None):
    """Referencia: en cada fold predice la media de motilidad de los OTROS folds (el test, la media de los 70)."""
    folds = load_folds() if folds is None else folds
    tv, test = folds[folds.split == "trainval"], folds[folds.split == "test"]
    cv = pd.DataFrame(index=tv.index, columns=PRED, dtype=float)
    for k in sorted(tv.fold.unique()):
        cv.loc[tv.fold == k, PRED] = tv.loc[tv.fold != k, TARGETS].mean().values
    te = pd.DataFrame(np.tile(tv[TARGETS].mean().values, (len(test), 1)), index=test.index, columns=PRED)
    return cv, te


def per_participant_error(cv, folds=None):
    folds = load_folds() if folds is None else folds
    return pd.Series(np.abs(cv.values - folds.loc[cv.index, TARGETS].values).mean(axis=1), index=cv.index)
