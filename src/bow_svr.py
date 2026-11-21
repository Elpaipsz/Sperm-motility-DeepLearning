"""Réplica del enfoque de motilitAI (Ottl et al., iScience 2022) sobre nuestras trayectorias y particiones.

motilitAI sigue células individuales (Trackpy), describe cada trayectoria, agrupa las trayectorias en un "vocabulario"
con k-means y describe cada video por un histograma de palabras (bolsa de palabras, BoW); luego un SVR lineal con el
parámetro C elegido por CV interna de 5 folds. Su mejor resultado (MAE 7,31, 3 folds de MediaEval) salió de una rejilla
sobre el tamaño del vocabulario. Aquí se usa UNA configuración fijada de antemano (vocabulario de 100 palabras,
asignación dura, frecuencias normalizadas) para no inflar el resultado, con nuestros descriptores por trayectoria
(los mismos 6 de la red de conjuntos) y el detector B.
El vocabulario y el SVR se ajustan SOLO con los participantes de entrenamiento de cada fold.
Uso: python src/bow_svr.py  ->  reports/bow_cv_pred_B_evisan_to_visem.csv, bow_test_pred_B_evisan_to_visem.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

sys.path.insert(0, str(Path(__file__).parent))
import track_set_net as T   # noqa: E402
from results_utils import PRED, TARGETS, load_folds   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP, K, SEED = "B_evisan_to_visem", 100, 42


def histograms(km, mu, sd, tracks, ids):
    H = np.zeros((len(ids), K))
    for r, i in enumerate(ids):
        lab = km.predict((tracks[i] - mu) / sd)
        H[r] = np.bincount(lab, minlength=K) / len(lab)
    return H


def fit_predict(tracks, tr_ids, Ytr, te_ids):
    X = np.concatenate([tracks[i] for i in tr_ids]); mu, sd = X.mean(0), X.std(0) + 1e-6
    rng = np.random.default_rng(SEED); X = X[rng.choice(len(X), min(len(X), 100_000), replace=False)]
    km = MiniBatchKMeans(n_clusters=K, random_state=SEED, n_init=3, batch_size=4096).fit((X - mu) / sd)
    Htr, Hte = histograms(km, mu, sd, tracks, tr_ids), histograms(km, mu, sd, tracks, te_ids)
    P = []
    for j in range(3):
        gs = GridSearchCV(make_pipeline(StandardScaler(), LinearSVR(random_state=SEED, max_iter=20000)),
                          {"linearsvr__C": [1e-3, 1e-2, 1e-1, 1, 10]}, cv=5, scoring="neg_mean_absolute_error")
        P.append(gs.fit(Htr, Ytr[:, j]).predict(Hte))
    P = np.clip(np.column_stack(P), 0, None)
    return P / P.sum(1, keepdims=True) * 100


def main():
    F = load_folds(); tv, te = F[F.split == "trainval"], F[F.split == "test"]
    table = T.build_track_table(EXP)
    tracks = {pid: T.track_matrix(g) for pid, g in table.groupby("ID")}
    cv = []
    for k in sorted(tv.fold.unique()):
        tr, va = tv[tv.fold != k], tv[tv.fold == k]
        cv.append(pd.DataFrame(fit_predict(tracks, tr.index.tolist(), tr[TARGETS].values, va.index.tolist()), index=va.index, columns=PRED).assign(fold=k))
        print("fold", k, "listo", flush=True)
    cv = pd.concat(cv)
    tep = pd.DataFrame(fit_predict(tracks, tv.index.tolist(), tv[TARGETS].values, te.index.tolist()), index=te.index, columns=PRED)
    cv.rename_axis("ID").reset_index().to_csv(ROOT / "reports" / f"bow_cv_pred_{EXP}.csv", index=False)
    tep.rename_axis("ID").reset_index().to_csv(ROOT / "reports" / f"bow_test_pred_{EXP}.csv", index=False)
    m = np.mean([np.abs(cv[cv.fold == k][PRED].values - tv.loc[cv[cv.fold == k].index, TARGETS].values).mean() for k in sorted(tv.fold.unique())])
    print(f"BoW + SVR · MAE CV {m:.2f} | test {np.abs(tep.values - te[TARGETS].values).mean():.2f}")


if __name__ == "__main__":
    main()
