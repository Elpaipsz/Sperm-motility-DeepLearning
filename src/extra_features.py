"""Características CASA adicionales por participante (ALH, BCF, giro medio, trayectorias "en el sitio").

Complementan motion_features.py con los parámetros de oscilación y giro que ese módulo no calcula:
  ALH  = amplitud del desplazamiento lateral de la cabeza: 2 x distancia media entre el camino crudo y el suavizado
  BCF  = frecuencia de cruce: veces por segundo que el camino crudo cruza al suavizado
  MAD  = giro medio absoluto del camino suavizado (grados por paso a 10 Hz)
  "en el sitio" = fracción de trayectorias con mucho camino (VCL alto) pero poco avance neto (LIN bajo)
Las unidades son píxeles (VISEM no trae escala µm/px). Misma corrección de deriva y longitud mínima de 1 s.
Uso: python src/extra_features.py <exp_detector>  ->  data/processed/features_extra_<exp>.csv (solo las nuevas)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent))
from motion_features import MIN_TRACK_S, SMOOTH_S, remove_drift   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def extra_metrics(track, fps):
    xy = track[["cx", "cy"]].values
    dur = track.t_s.values[-1] - track.t_s.values[0]
    if dur < MIN_TRACK_S or len(xy) < 8:
        return None
    k = max(3, int(round(SMOOTH_S * fps)))
    sm = np.column_stack([uniform_filter1d(xy[:, i], k, mode="nearest") for i in range(2)])
    dev = xy - sm
    alh = 2.0 * np.linalg.norm(dev, axis=1).mean()
    tang = np.gradient(sm, axis=0); tn = np.linalg.norm(tang, axis=1, keepdims=True)
    cross = (tang / np.maximum(tn, 1e-9))[:, 0] * dev[:, 1] - (tang / np.maximum(tn, 1e-9))[:, 1] * dev[:, 0]
    sg = np.sign(cross[np.abs(cross) > 0.05]); bcf = (np.diff(sg) != 0).sum() / dur if len(sg) > 1 else 0.0
    step = max(1, int(round(fps / 10)))
    sub = sm[::step]
    if len(sub) >= 4:
        ang = np.arctan2(np.diff(sub[:, 1]), np.diff(sub[:, 0]))
        turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi); mad = np.degrees(turn).mean()
    else:
        mad = 0.0
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    vcl, vsl = path / dur, np.linalg.norm(xy[-1] - xy[0]) / dur
    return dict(alh=alh, bcf=bcf, mad=mad, vcl=vcl, lin=vsl / vcl if vcl > 0 else 0.0)


def features_extra(df, fps):
    df = remove_drift(df)
    rows = [m for _, tr in df.groupby("track_id") if (m := extra_metrics(tr.sort_values("frame"), fps)) is not None]
    if not rows:
        return {}
    m = pd.DataFrame(rows)
    f = {}
    for c in ("alh", "bcf", "mad"):
        f[f"{c}_mean"], f[f"{c}_median"], f[f"{c}_p90"] = m[c].mean(), m[c].median(), np.percentile(m[c], 90)
    f["alh_over_vcl"] = (m.alh / m.vcl.clip(lower=1e-6)).median()
    f["frac_in_place"] = ((m.vcl > np.percentile(m.vcl, 50)) & (m.lin < 0.3)).mean()
    f["frac_mad_gt30"] = (m.mad > 30).mean()
    return f


def main():
    exp = sys.argv[1]
    manifest = pd.read_csv(ROOT / "data" / "processed" / "visem_gray" / "manifest.csv").set_index("ID")
    rows = []
    for f in sorted((ROOT / "data" / "processed" / "tracks" / exp).glob("*.csv.gz"), key=lambda p: int(p.name.split(".")[0])):
        pid = int(f.name.split(".")[0])
        rows.append(dict(ID=pid, **features_extra(pd.read_csv(f), manifest.loc[pid, "fps"])))
    out = ROOT / "data" / "processed" / f"features_extra_{exp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Guardado:", out, "|", len(rows), "participantes")


if __name__ == "__main__":
    main()
