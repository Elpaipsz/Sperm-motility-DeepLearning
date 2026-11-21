"""Características de movimiento por participante a partir de las trayectorias (src/track_videos.py).

Parámetros tipo CASA calculados por trayectoria, en píxeles y segundos reales (VISEM no trae la escala
µm/píxel, así que las velocidades son relativas, en px/s):
  VCL = longitud del camino / duración | VAP = igual sobre el camino suavizado (0,5 s)
  VSL = desplazamiento inicio-fin / duración | LIN = VSL/VCL | STR = VSL/VAP
Se corrige la deriva global (14 videos traen notas de 'drift'): en cada paso de frame se resta el
desplazamiento mediano de todas las trayectorias, que es el movimiento de la imagen, no de las células.
Uso: python src/motion_features.py <experimento_detector>   ->  data/processed/features_<exp>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_TRACK_S = 1.0     # trayectorias de menos de 1 s no sirven para medir movimiento
SMOOTH_S = 0.5
VSL_THRESHOLDS = (5, 10, 20, 40)   # px/s, para "fracción de trayectorias casi quietas"


def remove_drift(df):
    """Resta el desplazamiento mediano entre frames consecutivos (deriva de la muestra/cámara)."""
    df = df.sort_values(["track_id", "frame"]).copy()
    g = df.groupby("track_id")
    df["dx"], df["dy"], df["dframe"] = g.cx.diff(), g.cy.diff(), g.frame.diff()
    step = df[df.dframe == 1]
    if len(step) == 0:
        return df.drop(columns=["dx", "dy", "dframe"])
    med = step.groupby("frame")[["dx", "dy"]].median()
    n = step.groupby("frame").size()
    med[n < 3] = 0.0   # con menos de 3 trayectorias la mediana no es fiable
    full = med.reindex(range(int(df.frame.min()), int(df.frame.max()) + 1), fill_value=0.0).cumsum()
    df["cx"] -= full.loc[df.frame, "dx"].values
    df["cy"] -= full.loc[df.frame, "dy"].values
    return df.drop(columns=["dx", "dy", "dframe"])


def track_metrics(track, fps):
    t = track.t_s.values
    xy = track[["cx", "cy"]].values
    dur = t[-1] - t[0]
    if dur < MIN_TRACK_S:
        return None
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    k = max(3, int(round(SMOOTH_S * fps)))
    sm = np.column_stack([np.convolve(xy[:, i], np.ones(k) / k, mode="valid") for i in range(2)])
    vap = np.linalg.norm(np.diff(sm, axis=0), axis=1).sum() / dur
    vcl = path / dur
    vsl = np.linalg.norm(xy[-1] - xy[0]) / dur
    return dict(vcl=vcl, vap=vap, vsl=vsl, lin=vsl / vcl if vcl > 0 else 0.0, str_=vsl / vap if vap > 0 else 0.0)


def features_from_tracks(df, fps):
    n_frames = df.frame.nunique()
    feats = dict(det_per_frame=len(df) / max(n_frames, 1), n_tracks=df.track_id.nunique())
    df = remove_drift(df)
    rows = [m for _, tr in df.groupby("track_id") if (m := track_metrics(tr, fps)) is not None]
    feats["n_long_tracks"] = len(rows)
    if not rows:
        return feats
    m = pd.DataFrame(rows)
    for col in ("vcl", "vap", "vsl", "lin", "str_"):
        feats[f"{col}_mean"] = m[col].mean()
        feats[f"{col}_median"] = m[col].median()
    for q in (25, 75, 90):
        feats[f"vap_p{q}"] = np.percentile(m.vap, q)
        feats[f"vsl_p{q}"] = np.percentile(m.vsl, q)
    for thr in VSL_THRESHOLDS:
        feats[f"frac_vsl_lt{thr}"] = (m.vsl < thr).mean()
    feats["frac_lin_gt05"] = (m.lin > 0.5).mean()
    return feats


def main():
    exp = sys.argv[1]
    manifest = pd.read_csv(ROOT / "data" / "processed" / "visem_gray" / "manifest.csv").set_index("ID")
    rows = []
    for f in sorted((ROOT / "data" / "processed" / "tracks" / exp).glob("*.csv.gz"), key=lambda p: int(p.name.split(".")[0])):
        pid = int(f.name.split(".")[0])
        rows.append(dict(ID=pid, **features_from_tracks(pd.read_csv(f), manifest.loc[pid, "fps"])))
    out = ROOT / "data" / "processed" / f"features_{exp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Guardado:", out, "|", len(rows), "participantes")


if __name__ == "__main__":
    main()
