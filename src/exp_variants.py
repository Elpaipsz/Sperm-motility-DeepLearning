"""Experimentos de Deep Learning sobre la línea base de video (mismos folds, mismos clips, misma pérdida).

Cada variante reutiliza src/baseline_video.py cambiando UNA idea, para que la comparación sea limpia:

  resnet : ResNet18 preentrenada en ImageNet sobre 3 imágenes de movimiento por clip
           (media temporal, desviación temporal y diferencia absoluta media entre frames).
           Motivo: la CNN 3D sobreajusta (train KL ~0,04, val MAE ~11 pp con 56 participantes);
           se prueba partir de características ya aprendidas. OJO: afinar layer4 NO reduce los parámetros
           libres (8,4 M frente a 0,3 M de la CNN 3D); por eso existe resnet_probe.
  resnet_probe : igual, con el backbone congelado y solo la cabeza lineal entrenable (~1,5 mil parámetros):
           es la versión que sí prueba "pocos parámetros libres".
  reg3d  : la misma CNN 3D pero más pequeña y regularizada (canales a la mitad, dropout 0,5,
           Dropout3d entre bloques, weight decay 5e-2, inversión temporal y brillo aleatorio).
           Motivo: atacar el mismo sobreajuste sin cambiar de familia de modelo.

  r3d    : ResNet 3D de 18 capas (r3d_18) preentrenada en Kinetics-400 (video), afinando solo layer4 + cabeza,
           con 3 canales por frame (brillo normalizado, brillo menos media temporal, |diferencia entre frames|).
           Motivo: cerrar la comparación con un modelo de VIDEO preentrenado (la ResNet18 usa imágenes de ImageNet).

Uso: python src/exp_variants.py <resnet|resnet_probe|reg3d|r3d> [semilla]
Salida: reports/baseline_predictions_<variante>_s<semilla>.csv, models/<variante>_s<semilla>/fold*/history.csv
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision

import baseline_video as bv


def motion_images(clip):
    """clip (T,H,W) uint8 -> (3,H,W): media temporal, desviación temporal, |diferencia| media.
    El clip se normaliza en brillo (como en la línea base) pero NO se normaliza la magnitud de movimiento:
    esa magnitud es justo la señal que distingue una muestra móvil de una inmóvil."""
    x = torch.from_numpy(np.ascontiguousarray(clip)).float() / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    mean = x.mean(dim=0)
    std = x.std(dim=0) * 3.0
    diff = (x[1:] - x[:-1]).abs().mean(dim=0) * 3.0
    return torch.stack([mean, std, diff])


def r3d_channels(clip):
    """clip (T,H,W) uint8 -> (3,T,H,W): brillo normalizado, brillo menos media temporal, |diferencia entre frames|."""
    x = torch.from_numpy(np.ascontiguousarray(clip)).float() / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    d = (x[1:] - x[:-1]).abs() * 3.0
    return torch.stack([x, x - x.mean(dim=0, keepdim=True), torch.cat([d[:1], d])])


class R3DMotion(nn.Module):
    """r3d_18 (Kinetics-400). Solo se entrenan layer4 y la cabeza nueva."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith("layer4")
        net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 3))
        self.net = net

    def forward(self, x):
        return self.net(x)


class ResNetMotion(nn.Module):
    """ResNet18 (ImageNet) con cabeza nueva. train_layer4=True: se afina layer4 (~8,4 M de parámetros
    entrenables, MÁS que la CNN 3D de ~0,3 M). train_layer4=False: solo la cabeza lineal (~1,5 mil)."""
    train_layer4 = True

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        for name, p in net.named_parameters():
            p.requires_grad = self.train_layer4 and name.startswith("layer4")
        net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 3))
        self.net = net

    def forward(self, x):
        return self.net(x)


class ResNetProbe(ResNetMotion):
    """Ablación: backbone totalmente congelado, solo se entrena la capa lineal final."""
    train_layer4 = False


class Reg3D(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o, k, s):
            return nn.Sequential(nn.Conv3d(i, o, k, stride=s, padding=[v // 2 for v in k]), nn.BatchNorm3d(o),
                                 nn.ReLU(inplace=True), nn.Dropout3d(0.1))
        self.net = nn.Sequential(block(2, 8, (3, 5, 5), (1, 2, 2)), block(8, 16, (3, 3, 3), (2, 2, 2)),
                                 block(16, 32, (3, 3, 3), (2, 2, 2)), block(32, 64, (3, 3, 3), (2, 2, 2)),
                                 nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


def main():
    variant = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    bv.SEED = seed
    bv.OUT = bv.ROOT / "models" / f"{variant}_s{seed}"
    bv.SUFFIX = f"_{variant}_s{seed}"
    if variant == "resnet":
        bv.make_channels = motion_images
        bv.Baseline3D = ResNetMotion
        bv.LR = 3e-4
        bv.EPOCHS = 12
    elif variant == "resnet_probe":
        bv.make_channels = motion_images
        bv.Baseline3D = ResNetProbe
        bv.LR = 1e-3          # solo se entrena la cabeza lineal: un lr mayor que al afinar layer4
        bv.EPOCHS = 12
    elif variant == "r3d":
        bv.make_channels = r3d_channels
        bv.Baseline3D = R3DMotion
        bv.LR = 3e-4
        bv.EPOCHS = 12
    elif variant == "reg3d":
        bv.Baseline3D = Reg3D
        bv.WD, bv.AUG_EXTRA = 5e-2, True
    else:
        raise SystemExit("variante desconocida")
    n = sum(p.numel() for p in bv.Baseline3D().parameters() if p.requires_grad)
    print(f"{variant} semilla {seed}: {n:,} parámetros entrenables", flush=True)
    bv.main()


if __name__ == "__main__":
    main()
