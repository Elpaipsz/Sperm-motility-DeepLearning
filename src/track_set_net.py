"""Red de conjuntos (Deep Sets) sobre las trayectorias individuales: Deep Learning aplicado al método ganador.

Idea: cada participante es un CONJUNTO de trayectorias (cientos). Cada trayectoria se describe con 6 números
(VCL, VAP, VSL, LIN, STR, duración; mismas definiciones CASA que motion_features.py). Una MLP pequeña codifica cada
trayectoria, se agrupa con media y máximo (invariante al orden y al número de trayectorias) y una cabeza lineal da
los 3 porcentajes con softmax. Pérdida: KL contra la motilidad real (suma 100), igual que la CNN 3D.
A diferencia de Ridge, la red aprende por sí misma qué estadísticos del conjunto importan (no solo medias y
percentiles fijados a mano) y puede combinar velocidad y linealidad trayectoria a trayectoria.

Aumento de datos: en cada época se recorta una ventana temporal aleatoria (45-120 s) de cada video y se usa un
subconjunto de las trayectorias. Es el mismo participante visto con otro muestreo (hay ~70 participantes; el
conjunto de trayectorias cambia, la etiqueta no).

Protocolo idéntico al resto: folds de data/splits/folds.csv, 3 semillas, última época (sin elegir época con la
validación), test promediado entre los 5 modelos de fold.
Uso: python src/track_set_net.py <exp_detector> [semilla] [hidden]   ->  reports/baseline_predictions_trackset<tag>_s<semilla>.csv
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).parent))
from motion_features import MIN_TRACK_S, remove_drift, track_metrics   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["progressive", "non_progressive", "immotile"]
VIDEO_S = 120.0
EPOCHS, LR, WD = 300, 3e-3, 1e-2
DEVICE = "cpu"   # modelo diminuto: la CPU basta y evita competir con la GPU


def build_track_table(exp):
    """Una fila por trayectoria larga: id, inicio (s), duración y descriptores CASA. Se guarda una sola vez."""
    out = ROOT / "data" / "processed" / f"track_table_{exp}.csv.gz"
    if out.exists():
        return pd.read_csv(out)
    manifest = pd.read_csv(ROOT / "data" / "processed" / "visem_gray" / "manifest.csv").set_index("ID")
    rows = []
    for f in sorted((ROOT / "data" / "processed" / "tracks" / exp).glob("*.csv.gz")):
        pid = int(f.name.split(".")[0])
        df = remove_drift(pd.read_csv(f)); fps = manifest.loc[pid, "fps"]
        for _, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            m = track_metrics(tr, fps)
            if m is not None:
                rows.append(dict(ID=pid, t0=tr.t_s.iloc[0], dur=tr.t_s.iloc[-1] - tr.t_s.iloc[0], **m))
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False)
    return t


def track_matrix(t):
    """Transformación a escala logarítmica de lo que es asimétrico (velocidades, duración)."""
    return np.column_stack([np.log1p(t.vcl), np.log1p(t.vap), np.log1p(t.vsl), t.lin.clip(0, 1.5), t.str_.clip(0, 3),
                            np.log1p(t.dur)]).astype(np.float32)


class SetNet(nn.Module):
    def __init__(self, d_in=6, hidden=32, p_drop=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * hidden + 1, 3))

    def forward(self, x, mask, dens):
        # x: (B, N, d), mask: (B, N) 1 = trayectoria real, dens: (B, 1) log de trayectorias por segundo
        h = self.enc(x)
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(m == 0, -1e9).max(1).values
        return self.head(torch.cat([mean, mx, dens], dim=1))


def make_batch(ids, tracks, mu, sd, rng=None, train=False):
    """Arma un lote con relleno. En entrenamiento recorta una ventana temporal aleatoria y submuestrea."""
    xs, dens = [], []
    for i in ids:
        t = tracks[i]
        X, t0 = t["X"], t["t0"]
        if train:
            L = rng.uniform(45, VIDEO_S); s = rng.uniform(0, VIDEO_S - L)
            keep = (t0 >= s) & (t0 < s + L)
            if keep.sum() < 5:
                keep = np.ones(len(X), bool); L = VIDEO_S
            X = X[keep]
            if rng.random() < 0.5:   # además, submuestreo aleatorio de trayectorias
                X = X[rng.random(len(X)) < rng.uniform(0.5, 1.0)]
            if len(X) < 2:
                X = t["X"]; L = VIDEO_S
        else:
            L = VIDEO_S
        xs.append((X - mu) / sd); dens.append(np.log1p(len(X) / L * 10))
    n = max(len(x) for x in xs)
    B = np.zeros((len(xs), n, xs[0].shape[1]), np.float32); M = np.zeros((len(xs), n), np.float32)
    for k, x in enumerate(xs):
        B[k, :len(x)] = x; M[k, :len(x)] = 1
    return (torch.tensor(B, device=DEVICE), torch.tensor(M, device=DEVICE),
            torch.tensor(np.array(dens, np.float32)[:, None], device=DEVICE))


def train_fold(tr_ids, tracks, Y, seed, hidden, p_drop, log_ids=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    allx = np.concatenate([tracks[i]["X"] for i in tr_ids])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    net = SetNet(hidden=hidden, p_drop=p_drop).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tgt = torch.tensor(np.array([Y[i] for i in tr_ids]) / 100, dtype=torch.float32, device=DEVICE)
    hist = []
    for ep in range(EPOCHS):
        net.train()
        order = rng.permutation(len(tr_ids))
        for b in np.array_split(order, 4):   # minilotes de ~14 participantes
            x, m, d = make_batch([tr_ids[j] for j in b], tracks, mu, sd, rng, train=True)
            loss = Fn.kl_div(Fn.log_softmax(net(x, m, d), 1), tgt[b], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if log_ids is not None and (ep % 10 == 9 or ep == 0):
            hist.append(dict(epoch=ep + 1, train_kl=float(loss), val_mae=eval_mae(net, log_ids, tracks, Y, mu, sd)))
    return net, mu, sd, hist


@torch.no_grad()
def predict(net, ids, tracks, mu, sd):
    net.eval()
    x, m, d = make_batch(ids, tracks, mu, sd)
    return (torch.softmax(net(x, m, d), 1).cpu().numpy() * 100)


def eval_mae(net, ids, tracks, Y, mu, sd):
    return float(np.abs(predict(net, ids, tracks, mu, sd) - np.array([Y[i] for i in ids])).mean())


def main():
    exp = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    p_drop = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    tag = os.environ.get("SET_TAG", "")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    table = build_track_table(exp)
    folds = pd.read_csv(ROOT / "data" / "splits" / "folds.csv")
    Y = {r.ID: r[TARGETS].values.astype(float) for _, r in folds.iterrows()}
    tracks = {pid: dict(X=track_matrix(g), t0=g.t0.values) for pid, g in table.groupby("ID")}
    te_ids = folds[folds.split == "test"].ID.tolist()
    rows, hists = [], []
    for k in sorted(folds[folds.split == "trainval"].fold.unique()):
        tr = folds[(folds.split == "trainval") & (folds.fold != k)].ID.tolist()
        va = folds[(folds.split == "trainval") & (folds.fold == k)].ID.tolist()
        net, mu, sd, hist = train_fold(tr, tracks, Y, seed + int(k), hidden, p_drop, log_ids=va)
        for split, ids in (("cv", va), ("test", te_ids)):
            for i, p in zip(ids, predict(net, ids, tracks, mu, sd)):
                rows.append(dict(ID=i, split=split, fold=k, **{f"pred_{t}": v for t, v in zip(TARGETS, p)}))
        hists += [dict(fold=k, **h) for h in hist]
        print(f"fold {k}: MAE val última época {hist[-1]['val_mae']:.2f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "reports" / f"baseline_predictions_trackset{tag}_s{seed}.csv", index=False)
    pd.DataFrame(hists).to_csv(ROOT / "reports" / f"trackset_history{tag}_s{seed}.csv", index=False)
    cv = out[out.split == "cv"].set_index("ID")
    mae = np.abs(cv[[f"pred_{t}" for t in TARGETS]].values - folds.set_index("ID").loc[cv.index, TARGETS].values).mean()
    print(f"CV MAE (70) = {mae:.2f}")


if __name__ == "__main__":
    main()
