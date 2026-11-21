"""Línea base directa (objetivo 3): CNN 3D pequeña sobre clips de video -> % progresiva / no progresiva / inmóvil.

Entrada: clips de 32 frames (1,28 s a 25 fps) en gris de visem_gray/ (320x240), 2 canales:
  (1) frame normalizado por clip, (2) frame menos la media temporal del clip (quita el fondo estático y
  deja lo que se mueve). Salida: softmax de 3 clases = distribución de motilidad (suma 100 %).
Pérdida: KL(objetivo || predicción), coherente con que las tres cifras sumen 100.
Validación: 5 folds agrupados por participante (data/splits/folds.csv); el test (15) se predice con el
promedio de los 5 modelos de fold. Predicción por participante = promedio de sus clips.
Uso: python src/baseline_video.py   ->  reports/baseline_predictions.csv, models/baseline/fold*/history.csv
"""
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
GRAY = ROOT / "data" / "processed" / "visem_gray"
SEED = int(os.environ.get("BASE_SEED", 42))   # semillas extra para el ensamble (42 = corrida original)
SUFFIX = "" if SEED == 42 else f"_s{SEED}"
OUT = ROOT / "models" / f"baseline{SUFFIX}"
_, T, CROP, EPOCHS, CLIPS_PER_PART, BATCH, LR = 42, 32, 160, 15, 24, 16, 1e-3
TARGETS = ["progressive", "non_progressive", "immotile"]
WD = 1e-2            # decaimiento de pesos de AdamW (los experimentos de src/exp_variants.py lo cambian)
AUG_EXTRA = False    # True: además invierte el clip en el tiempo y varía el brillo (no altera la rapidez del movimiento)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_channels(clip):
    """clip: (T, H, W) uint8 -> (2, T, H, W) float32."""
    x = torch.from_numpy(np.ascontiguousarray(clip)).float() / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return torch.stack([x, x - x.mean(dim=0, keepdim=True)])


class ClipDataset(torch.utils.data.Dataset):
    """Clips aleatorios (train) o determinista (eval) de los participantes dados."""

    def __init__(self, ids, targets, train):
        self.ids, self.targets, self.train = list(ids), targets, train
        self.data = {i: [np.load(GRAY / f"{i}_w{w}.npy", mmap_mode="r") for w in range(4)] for i in self.ids}
        if train:
            self.index = [(i, None, None) for i in self.ids for _ in range(CLIPS_PER_PART)]
        else:   # 5 clips por ventana, equiespaciados
            self.index = [(i, w, s) for i in self.ids for w in range(4) for s in range(0, 750 - T, 150)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        pid, w, s = self.index[k]
        if self.train:
            w, s = random.randrange(4), random.randrange(750 - T)
        clip = self.data[pid][w][s:s + T]
        if self.train:
            y0, x0 = random.randrange(240 - CROP + 1), random.randrange(320 - CROP + 1)
            clip = clip[:, y0:y0 + CROP, x0:x0 + CROP]
            if random.random() < 0.5: clip = clip[:, :, ::-1]
            if random.random() < 0.5: clip = clip[:, ::-1, :]
            if AUG_EXTRA:
                if random.random() < 0.5: clip = clip[::-1]
                clip = np.clip(clip.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-12, 12), 0, 255)
        return make_channels(clip), torch.tensor(self.targets[pid], dtype=torch.float32), pid


class Baseline3D(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o, k, s):
            return nn.Sequential(nn.Conv3d(i, o, k, stride=s, padding=[x // 2 for x in k]), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(block(2, 16, (3, 5, 5), (1, 2, 2)), block(16, 32, (3, 3, 3), (2, 2, 2)),
                                 block(32, 64, (3, 3, 3), (2, 2, 2)), block(64, 128, (3, 3, 3), (2, 2, 2)),
                                 nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        return self.net(x)


def kl_loss(logits, target):
    return F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")


@torch.no_grad()
def predict(model, ids, targets, device):
    model.eval()
    ds = ClipDataset(ids, targets, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    probs = {i: [] for i in ids}
    for x, _, pid in dl:
        with torch.autocast("cuda"):
            p = F.softmax(model(x.to(device)), dim=1).float().cpu().numpy()
        for i, row in zip(pid.tolist(), p):
            probs[i].append(row)
    return {i: np.mean(v, axis=0) * 100 for i, v in probs.items()}


def run_fold(fold, folds, targets, device):
    seed_all(SEED + fold)
    tr_ids = folds[(folds.split == "trainval") & (folds.fold != fold)].ID.tolist()
    va_ids = folds[(folds.split == "trainval") & (folds.fold == fold)].ID.tolist()
    model = Baseline3D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler()
    dl = torch.utils.data.DataLoader(ClipDataset(tr_ids, targets, True), batch_size=BATCH, shuffle=True, num_workers=0)
    out = OUT / f"fold{fold}"; out.mkdir(parents=True, exist_ok=True)
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y, _ in dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda"):
                loss = kl_loss(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x); n += len(x)
        sched.step()
        pred = predict(model, va_ids, targets, device)
        mae = np.mean([abs(pred[i] - np.array(targets[i]) * 100).mean() for i in va_ids])
        hist.append(dict(epoch=ep, train_kl=tot / n, val_mae_pp=mae, seconds=time.time() - t0))
        print(f"fold {fold} ep {ep:02d} train_kl {tot / n:.4f} val_MAE {mae:.2f} pp", flush=True)
        torch.save(model.state_dict(), out / "last.pt")
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    # Se usa SIEMPRE la última época: elegir la mejor época con el mismo fold que se reporta sesgaría el MAE.
    return hist[-1]["val_mae_pp"]


def main():
    device = torch.device("cuda")
    folds = pd.read_csv(ROOT / "data" / "splits" / "folds.csv")
    targets = {int(r.ID): [r.progressive / 100, r.non_progressive / 100, r.immotile / 100] for r in folds.itertuples()}
    rows = []
    for fold in range(5):
        run_fold(fold, folds, targets, device)
        model = Baseline3D().to(device); model.load_state_dict(torch.load(OUT / f"fold{fold}" / "last.pt"))
        va_ids = folds[(folds.split == "trainval") & (folds.fold == fold)].ID.tolist()
        te_ids = folds[folds.split == "test"].ID.tolist()
        for split, ids in (("cv", va_ids), ("test", te_ids)):
            for i, p in predict(model, ids, targets, device).items():
                rows.append(dict(ID=i, split=split, fold=fold, **{f"pred_{k}": v for k, v in zip(TARGETS, p)}))
    pd.DataFrame(rows).to_csv(ROOT / "reports" / f"baseline_predictions{SUFFIX}.csv", index=False)


if __name__ == "__main__":
    main()
