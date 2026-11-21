"""Validación cruzada ANIDADA para la red de conjuntos (src/track_set_net.py).

Para cada fold externo k: la configuración (tamaño oculto, dropout) se elige con una CV interna sobre los otros 4
folds (el fold k no interviene en la elección); luego se reentrena con esos 4 folds y se predice el fold k y el test.
Así la CV externa es una estimación sin sesgo de selección del procedimiento "ajustar y entrenar".
Se usan 100 épocas (la red converge hacia la 50, ver el Anexo C del cuaderno) para que la búsqueda sea viable.
Uso: python src/track_set_nested.py <exp_detector> <fold_externo>  ->  reports/nested_trackset_fold<k>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import track_set_net as T   # noqa: E402

torch.set_num_threads(1)
T.EPOCHS = 100
GRID = [(h, p) for h in (16, 32, 64) for p in (0.2, 0.5)]
SEED = 42


def main():
    exp, k = sys.argv[1], int(sys.argv[2])
    table = T.build_track_table(exp)
    folds = pd.read_csv(T.ROOT / "data" / "splits" / "folds.csv")
    Y = {r.ID: r[T.TARGETS].values.astype(float) for _, r in folds.iterrows()}
    tracks = {pid: dict(X=T.track_matrix(g), t0=g.t0.values) for pid, g in table.groupby("ID")}
    tv = folds[folds.split == "trainval"]
    outer_tr, outer_va = tv[tv.fold != k].ID.tolist(), tv[tv.fold == k].ID.tolist()
    te_ids = folds[folds.split == "test"].ID.tolist()
    inner = {}
    for h, p in GRID:
        maes = []
        for j in sorted(set(tv[tv.fold != k].fold)):
            tr = tv[(tv.fold != k) & (tv.fold != j)].ID.tolist(); va = tv[tv.fold == j].ID.tolist()
            net, mu, sd, _ = T.train_fold(tr, tracks, Y, SEED + j, h, p)
            maes.append(T.eval_mae(net, va, tracks, Y, mu, sd))
        inner[(h, p)] = float(np.mean(maes))
        print(f"fold externo {k} · hidden {h} drop {p}: MAE interno {inner[(h, p)]:.2f}", flush=True)
    best = min(inner, key=inner.get)
    net, mu, sd, _ = T.train_fold(outer_tr, tracks, Y, SEED + k, *best)
    rows = []
    for split, ids in (("cv", outer_va), ("test", te_ids)):
        for i, pr in zip(ids, T.predict(net, ids, tracks, mu, sd)):
            rows.append(dict(ID=i, split=split, fold=k, hidden=best[0], drop=best[1],
                             **{f"pred_{t}": v for t, v in zip(T.TARGETS, pr)}))
    pd.DataFrame(rows).to_csv(T.ROOT / "reports" / f"nested_trackset_fold{k}.csv", index=False)
    pd.DataFrame([dict(fold=k, hidden=h, drop=p, inner_mae=m) for (h, p), m in inner.items()]) \
        .to_csv(T.ROOT / "reports" / f"nested_trackset_inner_fold{k}.csv", index=False)
    print(f"fold externo {k}: elegido hidden={best[0]} drop={best[1]}")


if __name__ == "__main__":
    main()
