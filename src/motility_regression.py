"""Regresor de motilidad a partir de características de movimiento (objetivo 2).

Entrada: data/processed/features_<exp>.csv (src/motion_features.py) + data/splits/folds.csv.
Modelos: media del entrenamiento (referencia trivial), Ridge, Random Forest, Gradient Boosting.
Protocolo: validación cruzada de 5 folds agrupada POR PARTICIPANTE sobre los 70 de trainval (los folds
vienen de folds.csv, no se regeneran). Se elige el modelo por MAE medio en CV; el test (15 participantes)
se evalúa UNA vez con el modelo elegido, reentrenado con los 70.
Salida: tres porcentajes que suman 100 (se recortan a >= 0 y se renormalizan).
Uso: python src/motility_regression.py <experimento_detector>  ->  reports/regression_*.csv
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGETS = ["progressive", "non_progressive", "immotile"]

MODELS = {
    "media_entrenamiento": None,
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                          subsample=0.8, random_state=SEED),
}


def fit_predict(name, Xtr, Ytr, Xte):
    """Entrena un regresor por objetivo y devuelve (n, 3) normalizado a suma 100."""
    if name == "media_entrenamiento":
        P = np.tile(Ytr.mean(axis=0), (len(Xte), 1))
    else:
        P = np.column_stack([MODELS[name]().fit(Xtr, Ytr[:, k]).predict(Xte) for k in range(3)])
    P = np.clip(P, 0, None)
    return P / P.sum(axis=1, keepdims=True) * 100


def mae(P, Y):
    return np.abs(P - Y).mean(axis=0)   # por variable


def main():
    exp = sys.argv[1]
    random.seed(SEED); np.random.seed(SEED)
    folds = pd.read_csv(ROOT / "data" / "splits" / "folds.csv")
    feats = pd.read_csv(ROOT / "data" / "processed" / f"features_{exp}.csv")
    d = folds.merge(feats, on="ID")
    cols = [c for c in feats.columns if c != "ID"]
    tv, te = d[d.split == "trainval"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)

    rows, cv_pred = [], {}
    for name in MODELS:
        fold_mae, P_all = [], np.zeros((len(tv), 3))
        for k in sorted(tv.fold.unique()):
            tr, va = tv[tv.fold != k], tv[tv.fold == k]
            P = fit_predict(name, tr[cols].values, tr[TARGETS].values, va[cols].values)
            P_all[va.index] = P
            fold_mae.append(mae(P, va[TARGETS].values).mean())
        cv_pred[name] = P_all
        per_var = mae(P_all, tv[TARGETS].values)
        rows.append(dict(model=name, cv_mae_mean=np.mean(fold_mae), cv_mae_std=np.std(fold_mae),
                         mae_prog=per_var[0], mae_nonprog=per_var[1], mae_immot=per_var[2],
                         rmse=np.sqrt(((P_all - tv[TARGETS].values) ** 2).mean())))
    cv = pd.DataFrame(rows).round(2)
    print(cv.to_string(index=False))
    cv.to_csv(ROOT / "reports" / f"regression_cv_{exp}.csv", index=False)

    best = cv[cv.model != "media_entrenamiento"].sort_values("cv_mae_mean").iloc[0].model
    print("\nModelo elegido por CV:", best)
    res = []
    for name in ("media_entrenamiento", best):
        P = fit_predict(name, tv[cols].values, tv[TARGETS].values, te[cols].values)
        res.append(dict(model=name, test_mae=mae(P, te[TARGETS].values).mean(),
                        test_rmse=np.sqrt(((P - te[TARGETS].values) ** 2).mean()),
                        mae_prog=mae(P, te[TARGETS].values)[0]))
        if name == best:
            pd.DataFrame(P, columns=[f"pred_{t}" for t in TARGETS]).assign(ID=te.ID.values) \
                .to_csv(ROOT / "reports" / f"regression_test_pred_{exp}.csv", index=False)
    out = pd.DataFrame(res).round(2)
    print("\nTEST (una sola evaluación):\n", out.to_string(index=False))
    out.to_csv(ROOT / "reports" / f"regression_test_{exp}.csv", index=False)
    pd.DataFrame(cv_pred[best], columns=[f"pred_{t}" for t in TARGETS]).assign(ID=tv.ID.values, fold=tv.fold.values) \
        .to_csv(ROOT / "reports" / f"regression_cv_pred_{exp}.csv", index=False)


if __name__ == "__main__":
    main()
