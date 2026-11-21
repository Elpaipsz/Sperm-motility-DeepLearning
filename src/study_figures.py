"""Figuras de la guía de estudio: conceptos ilustrados y gráficas con los datos reales del proyecto.

Uso: python src/study_figures.py  ->  reports/figures/estudio/*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from results_utils import TARGETS, load_cnn_predictions, load_folds, load_regression_predictions, per_participant_error   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "estudio"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
V, M, O, G, T, B = "#5B3FD0", "#B3367B", "#E2702B", "#94A3B8", "#0E8C7F", "#2563EB"


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)


def box(ax, x, y, w, h, text, color, fc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", ec=color, fc=fc, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#121A2B" if fc == "white" else "white", wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#566174"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=1.4))


# 1. flujo del sistema
fig, ax = plt.subplots(figsize=(10, 2.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2)
steps = [("Video\n640×480, 50 fps", G), ("Detector\nYOLO11n", V), ("Seguimiento\nByteTrack", B), ("24 descriptores\npor participante", T), ("Ridge + red de\nconjuntos", M), ("% progresiva,\nno progresiva,\ninmóvil", O)]
for k, (t, c) in enumerate(steps):
    box(ax, 0.1 + k * 1.65, 0.45, 1.4, 1.1, t, c, fc=c if k == 5 else "white")
    if k < 5: arrow(ax, 1.5 + k * 1.65, 1.0, 1.75 + k * 1.65, 1.0)
ax.text(5, 0.12, "Línea base de comparación: CNN 3D que va directo del video a los porcentajes (sin seguir células)", ha="center", fontsize=9, color="#566174")
save(fig, "e01_flujo.png")

# 2. partición: fuga por frame vs por participante
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
rng = np.random.default_rng(0)
ax = axes[0]; ax.set_title("Mal: dividir por frame", fontsize=10, color="#B3361F")
for p in range(4):
    for f in range(12):
        c = O if rng.random() < 0.3 else V
        ax.add_patch(Rectangle((f, 3 - p), 0.9, 0.8, color=c))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
ax.text(6, -0.35, "frames del mismo participante en entrenamiento (violeta) y prueba (naranja): casi idénticos → fuga", ha="center", fontsize=8)
ax = axes[1]; ax.set_title("Bien: dividir por participante", fontsize=10, color="#1F7A5A")
for p in range(4):
    c = O if p == 3 else V
    for f in range(12): ax.add_patch(Rectangle((f, 3 - p), 0.9, 0.8, color=c))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
ax.text(6, -0.35, "todos los frames de un participante quedan en el mismo conjunto", ha="center", fontsize=8)
save(fig, "e02_particion.png")

# 3. validación cruzada de 5 folds + test
fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off"); ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.6)
for r in range(5):
    for k in range(5):
        ax.add_patch(Rectangle((k * 1.6 + 0.2, 4.4 - r), 1.5, 0.8, color=O if k == r else V, alpha=0.9))
    ax.text(8.4, 4.8 - r, f"ronda {r + 1}: valida el fold {r}", va="center", fontsize=8.5)
ax.add_patch(Rectangle((0.2, -0.05), 7.9, 0.35, color="#121A2B")); ax.text(4.15, 0.12, "test fijo de 15 participantes: se mira una sola vez al final", ha="center", va="center", color="white", fontsize=8.5)
ax.text(4.15, 5.35, "70 participantes en 5 folds (violeta entrena, naranja valida)", ha="center", fontsize=9)
save(fig, "e03_cv.png")

# 4. IoU
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
for ax, (dx, dy) in zip(axes, [(0.1, 0.05), (0.45, 0.3), (0.9, 0.9)]):
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=T, lw=2.2)); ax.add_patch(Rectangle((dx, dy), 1, 1, fill=False, ec=M, lw=2.2))
    ix = max(0, 1 - dx); iy = max(0, 1 - dy); inter = ix * iy; iou = inter / (2 - inter)
    if inter > 0: ax.add_patch(Rectangle((dx, dy), ix, iy, color=V, alpha=0.3))
    ax.set_xlim(-0.2, 2.1); ax.set_ylim(-0.2, 2.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"IoU = {iou:.2f}  →  {'acierto' if iou >= 0.5 else 'fallo'} (umbral 0,5)", fontsize=9)
fig.suptitle("IoU = área de intersección / área de unión (verde: caja real; magenta: caja predicha)", fontsize=10)
save(fig, "e04_iou.png")

# 5. sobreajuste, subajuste, convergencia
ep = np.arange(1, 51)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
cases = [("Subajuste", 1.0 - 0.1 * (1 - np.exp(-ep / 10)), 1.05 - 0.1 * (1 - np.exp(-ep / 10))),
         ("Buen ajuste: converge", 1.0 * np.exp(-ep / 12) + 0.25, 1.0 * np.exp(-ep / 12) + 0.32),
         ("Sobreajuste", 1.0 * np.exp(-ep / 10) + 0.1, 0.6 * np.exp(-ep / 6) + 0.35 + 0.006 * ep)]
for ax, (t, tr, va) in zip(axes, cases):
    ax.plot(ep, tr, color=B, label="entrenamiento"); ax.plot(ep, va, color=O, label="validación"); ax.set_title(t, fontsize=10); ax.set_xlabel("época")
axes[0].set_ylabel("pérdida"); axes[0].legend(fontsize=8)
save(fig, "e05_ajuste.png")

# 6. softmax y KL
z = np.array([1.2, 0.9, 1.0]); p = np.exp(z) / np.exp(z).sum(); q = np.array([0.33, 0.30, 0.37])
kl = float((q * np.log(q / p)).sum())
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
axes[0].bar(["progresiva", "no prog.", "inmóvil"], z, color=G); axes[0].set_title("Salida cruda de la red (logits)", fontsize=10)
x = np.arange(3); axes[1].bar(x - 0.2, q * 100, 0.4, color=T, label="real (laboratorio)"); axes[1].bar(x + 0.2, p * 100, 0.4, color=M, label="softmax (predicha)")
axes[1].set_xticks(x); axes[1].set_xticklabels(["progresiva", "no prog.", "inmóvil"]); axes[1].set_ylabel("%"); axes[1].legend(fontsize=8)
axes[1].set_title(f"Softmax convierte en porcentajes que suman 100; KL = {kl:.4f}".replace(".", ","), fontsize=10)
save(fig, "e06_softmax_kl.png")

# 7. tasa de aprendizaje con coseno
e = np.arange(0, 301); lr = 3e-3 * 0.5 * (1 + np.cos(np.pi * e / 300))
fig, ax = plt.subplots(figsize=(6, 2.4)); ax.plot(e, lr, color=V); ax.set_xlabel("época"); ax.set_ylabel("tasa de aprendizaje")
ax.set_title("Decaimiento coseno (red de conjuntos: 3·10⁻³ → 0 en 300 épocas)", fontsize=10)
save(fig, "e07_coseno.png")

# 8. VCL, VSL, LIN
t = np.linspace(0, 1, 60); xs = 10 + 160 * t + 6 * np.sin(18 * t); ys = 20 + 50 * t + 10 * np.sin(11 * t)
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
ax = axes[0]; ax.plot(xs, ys, color=O, lw=2, label="camino recorrido (VCL)"); ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], "--", color="#121A2B", label="avance neto (VSL)")
path = np.sum(np.hypot(np.diff(xs), np.diff(ys))); net = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
ax.set_title(f"Progresiva: LIN = VSL / VCL = {net / path:.2f}".replace(".", ","), fontsize=10); ax.legend(fontsize=8); ax.axis("equal"); ax.axis("off")
th = np.linspace(0, 6 * np.pi, 400); xs2 = 60 + 2.5 * th + 14 * np.cos(th) + 1.2 * np.sin(7 * th); ys2 = 50 + 12 * np.sin(th) + 1.2 * np.cos(7 * th)
p2 = np.sum(np.hypot(np.diff(xs2), np.diff(ys2))); n2 = np.hypot(xs2[-1] - xs2[0], ys2[-1] - ys2[0])
ax = axes[1]; ax.plot(xs2, ys2, color=M, lw=2); ax.plot([xs2[0], xs2[-1]], [ys2[0], ys2[-1]], "--", color="#121A2B")
ax.set_title(f"No progresiva: mucho camino, poco avance (LIN = {n2 / p2:.2f})".replace(".", ","), fontsize=10); ax.axis("equal"); ax.axis("off")
save(fig, "e08_vcl_vsl.png")

# 9. Deep Sets
fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
for k in range(4):
    y = 2.5 - k * 0.65
    box(ax, 0.1, y - 0.25, 1.5, 0.5, f"trayectoria {k + 1}" if k < 3 else "trayectoria n", G, fs=8)
    box(ax, 2.2, y - 0.25, 1.6, 0.5, "φ: red 6→32→32", V, fs=8); arrow(ax, 1.6, y, 2.2, y); arrow(ax, 3.8, y, 4.6, 1.5)
box(ax, 4.6, 1.1, 1.6, 0.8, "media y máximo\n(sin orden)", B, fs=8.5); arrow(ax, 6.2, 1.5, 6.8, 1.5)
box(ax, 6.8, 1.1, 1.4, 0.8, "ρ: lineal +\nsoftmax", M, fs=8.5); arrow(ax, 8.2, 1.5, 8.6, 1.5)
box(ax, 8.6, 1.1, 1.3, 0.8, "3 porcentajes\n(suman 100)", O, fc=O, fs=8.5)
ax.text(5.4, 2.55, "+ densidad de trayectorias", fontsize=8, color="#566174", ha="center")
save(fig, "e09_deepsets.png")

# ---- figuras con datos reales
F = load_folds()
r = load_regression_predictions("B_evisan_to_visem")
s = load_cnn_predictions([ROOT / "reports" / f"baseline_predictions_trackset_s{k}.csv" for k in (42, 7, 123)])
c = load_cnn_predictions([ROOT / "reports" / "baseline_predictions.csv", ROOT / "reports" / "baseline_predictions_s7.csv", ROOT / "reports" / "baseline_predictions_s123.csv"])
fin = ((r[0] + s[0]) / 2, (r[1] + s[1]) / 2)
E = lambda p: pd.concat([per_participant_error(p[0], F), per_participant_error(p[1], F)])
ec, ef = E(c), E(fin)

# 10. bootstrap de la diferencia pareada
d = (ec - ef).values; rngb = np.random.default_rng(42); bs = np.array([d[rngb.integers(0, len(d), len(d))].mean() for _ in range(10000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(8, 2.8)); ax.hist(bs, bins=60, color=V, alpha=0.8)
ax.axvline(lo, color=O, ls="--"); ax.axvline(hi, color=O, ls="--"); ax.axvline(0, color="#121A2B", lw=1)
ax.set_title(f"Bootstrap: error(CNN 3D) − error(sistema final), 10.000 remuestreos. Media {d.mean():.2f}; IC 95 % [{lo:.2f}; {hi:.2f}]", fontsize=9.5)
ax.set_xlabel("diferencia de MAE (puntos porcentuales)"); ax.text(0.05, ax.get_ylim()[1] * 0.8, "el 0 queda fuera\ndel intervalo →\ndiferencia real", fontsize=8.5)
save(fig, "e10_bootstrap.png")

# 11. contracción hacia la media
P = pd.concat([fin[0], fin[1]]); y = F.loc[P.index, "progressive"]; yp = P["pred_progressive"]
fig, ax = plt.subplots(figsize=(5.2, 4)); ax.scatter(y, yp, s=16, color=V, alpha=0.8); ax.plot([0, 80], [0, 80], "--", color=G, label="predicción perfecta")
m, b0 = np.polyfit(y, yp, 1); xx = np.array([0, 80]); ax.plot(xx, m * xx + b0, color=O, label=f"ajuste: pendiente {m:.2f}")
ax.set_xlabel("progresiva real (%)"); ax.set_ylabel("progresiva estimada (%)"); ax.legend(fontsize=8)
ax.set_title("Contracción hacia la media (85 fuera de muestra)", fontsize=10)
save(fig, "e11_contraccion.png")

# 12. intervalo conformal
cvi = fin[0].index; res = np.abs(fin[0].values - F.loc[cvi, TARGETS].values)[:, 0]; n = len(res)
qh = np.quantile(res, min(1, np.ceil((n + 1) * 0.9) / n))
fig, ax = plt.subplots(figsize=(8, 2.8)); ax.hist(res, bins=25, color=T, alpha=0.85); ax.axvline(qh, color=O, ls="--")
ax.set_title(f"Errores de CV en progresiva (70): el 90 % queda por debajo de {qh:.1f} pp → intervalo de ±{qh:.1f}", fontsize=10)
ax.set_xlabel("error absoluto (pp)"); ax.set_ylabel("participantes")
save(fig, "e12_conformal.png")

# 13. matriz de confusión (límite OMS)
yt = (F["progressive"] < 32).astype(int); ypr = (P["pred_progressive"] < 32).astype(int).reindex(F.index)
cm = np.array([[((yt == a) & (ypr == b)).sum() for b in (0, 1)] for a in (0, 1)])
fig, ax = plt.subplots(figsize=(4.4, 3.6)); ax.imshow(cm, cmap="Purples")
for i in range(2):
    for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="white" if cm[i, j] > 30 else "#121A2B")
ax.set_xticks([0, 1]); ax.set_xticklabels(["predice normal", "predice bajo"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["real normal", "real bajo"])
ax.set_title("Progresiva < 32 % (85 participantes)", fontsize=10)
save(fig, "e13_confusion.png")
print("figuras en", OUT, "| bootstrap:", round(d.mean(), 2), round(lo, 2), round(hi, 2), "| pendiente:", round(m, 2), "| conformal prog:", round(qh, 1), "| matriz:", cm.tolist())
