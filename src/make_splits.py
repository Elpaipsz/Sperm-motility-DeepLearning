"""Genera data/splits/folds.csv: split por PARTICIPANTE (nunca por frame o clip).

Diseño:
- Test fijo de 15 participantes (4 con cajas de VISEM-Tracking, 11 sin), nunca usados para entrenar
  ni el detector ni el regresor de motilidad.
- Los 70 restantes se reparten en 5 folds agrupados por participante.
- Todo estratificado por motilidad progresiva (la variable más variable), con semilla fija.
- Los participantes con cajas (VISEM-Tracking) se reparten parejo entre folds: el detector se
  entrena con los folds 1-4 y valida con el fold 0.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_FOLDS = 5
N_TEST_ANNOTATED = 4
N_TEST_OTHER = 11

ROOT = Path(__file__).resolve().parents[1]
VISEM = ROOT / "visem-dataset"
TRACKING = ROOT / "VISEM-Tracking" / "extracted" / "VISEM_Tracking_Train_v4" / "Train"


def load_table():
    sem = pd.read_csv(VISEM / "semen_analysis_data.csv", sep=";", decimal=",", encoding="utf-8-sig")
    vid = pd.read_csv(VISEM / "videos.csv", sep=";", encoding="utf-8-sig")
    df = sem[["ID", "Progressive motility (%)", "Non progressive sperm motility (%)", "Immotile sperm (%)"]]
    df.columns = ["ID", "progressive", "non_progressive", "immotile"]
    df = df.merge(vid, on="ID")
    df["evaluator"] = df["video"].str.extract(r"_([A-Za-z]+)\.avi$")
    df["quality_note"] = df["video"].str.extract(r"_(minor drift|drift|lots of debris)_", expand=False).fillna("")
    annotated = {int(p.name) for p in TRACKING.iterdir() if p.is_dir()}
    df["has_tracking"] = df["ID"].isin(annotated)
    return df


def pick_stratified(df, n, rng):
    """Elige n filas repartidas a lo largo del rango de motilidad progresiva (un bloque -> una fila)."""
    ordered = df.sort_values("progressive")
    blocks = np.array_split(np.arange(len(ordered)), n)
    return [ordered.index[rng.choice(b)] for b in blocks]


def deal_to_folds(df, rng, preference):
    """Reparte df en N_FOLDS por bloques de motilidad; el resto va a los folds de `preference`."""
    ordered = df.sort_values("progressive")
    out = {}
    idx = list(ordered.index)
    full = len(idx) // N_FOLDS * N_FOLDS
    for s in range(0, full, N_FOLDS):
        for i, f in zip(idx[s:s + N_FOLDS], rng.permutation(N_FOLDS)):
            out[i] = int(f)
    rest = idx[full:]
    for i, f in zip(rest, rng.permutation(preference[:len(rest)])):
        out[i] = int(f)
    return out


def main():
    rng = np.random.default_rng(SEED)
    df = load_table()
    df["split"] = "trainval"
    df["fold"] = -1

    test_idx = pick_stratified(df[df.has_tracking], N_TEST_ANNOTATED, rng)
    test_idx += pick_stratified(df[~df.has_tracking], N_TEST_OTHER, rng)
    df.loc[test_idx, "split"] = "test"

    tv = df[df.split == "trainval"]
    ann, oth = tv[tv.has_tracking], tv[~tv.has_tracking]
    # Los sobrantes de anotados van primero al fold 0 (validación del detector); los demás completan 14 por fold.
    for i, f in deal_to_folds(ann, rng, [0, 1, 2, 3, 4]).items():
        df.loc[i, "fold"] = f
    pref = [f for f in range(N_FOLDS) if (df[(df.fold == f) & df.has_tracking].shape[0]) <= 3] + list(range(N_FOLDS))
    for i, f in deal_to_folds(oth, rng, pref).items():
        df.loc[i, "fold"] = f

    df["detector_role"] = ""
    ann_mask = df.has_tracking
    df.loc[ann_mask & (df.split == "trainval") & (df.fold == 0), "detector_role"] = "val"
    df.loc[ann_mask & (df.split == "trainval") & (df.fold != 0), "detector_role"] = "train"
    df.loc[ann_mask & (df.split == "test"), "detector_role"] = "test"

    out = ROOT / "data" / "splits" / "folds.csv"
    df.sort_values("ID").to_csv(out, index=False)
    print("Guardado:", out)
    return df


if __name__ == "__main__":
    main()
