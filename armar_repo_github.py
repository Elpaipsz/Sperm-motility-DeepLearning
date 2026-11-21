"""Arma la carpeta que se sube al repositorio de GitHub: entrega/subir_a_github/.

Uso:
    python armar_repo_github.py

Copia, desde una lista explícita (no con comodines sobre la raíz), solo lo que debe estar en el repositorio,
reorganiza el material de estudio en material_estudio/ y escribe un README pensado para GitHub. Antes de terminar
revisa que no haya claves, videos, pesos de modelos ni archivos de más de 50 MB. Además deja dos archivos junto a
la carpeta: entrega/INSTRUCCIONES_SUBIDA_GITHUB.md (pasos para quien haga el push) y entrega/MANIFIESTO_SUBIDA.txt.

No toca los originales: solo crea o regenera entrega/subir_a_github/, que está ignorada por git.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTREGA = ROOT / "entrega"
DEST = ENTREGA / "subir_a_github"
REPO_URL = "https://github.com/Elpaipsz/Sperm-motility-DeepLearning"
MAX_MB = 50

# archivos de la raíz (lista explícita)
RAIZ = ["requirements.txt", ".gitignore", "estimacion_motilidad_visem.ipynb", "AGENTS.md", "empaquetar_entrega.py", "armar_repo_github.py"]
# documentos de reports/ (los CSV y las figuras se agregan aparte)
REPORTS = ["informe_final.pdf", "informe_final.docx", "presentacion.pdf", "presentacion.html", "documento_tecnico.pdf", "documento_tecnico.docx"]
# material interno de estudio: va a material_estudio/ (solo la última versión del apoyo para tableta)
ESTUDIO = {"guion_presentacion.docx": "guion_presentacion.docx", "guia_estudio.pdf": "guia_estudio.pdf",
           "guia_estudio.docx": "guia_estudio.docx", "apoyo_tableta_v4.pdf": "apoyo_tableta.pdf"}

PROHIBIDO_EXT = {".pt", ".pth", ".joblib", ".avi", ".mp4", ".wmv", ".npy", ".env", ".key", ".zip"}
PROHIBIDO_NOMBRE = ("reference-notebooks", "sperm-detection-project")
CLAVE = re.compile(r"(api[_-]?key|secret|token|password)\w*\s*[=:]\s*[\"'][A-Za-z0-9_\-]{12,}[\"']", re.I)   # asignación de una clave real
CODIGO = {".py", ".ipynb"}   # en el código no debe aparecer «roboflow» (la clave ajena venía de ahí)
TEXTOS = {".py", ".md", ".txt", ".ipynb", ".html", ".csv", ".json"}

TREE = """```
.
├── README.md
├── estimacion_motilidad_visem.ipynb   cuaderno principal, ejecutado (secciones 1 a 7 y anexos A a F)
├── requirements.txt                   versiones exactas del entorno
├── src/                               scripts: preparar datos, entrenar, evaluar, generar figuras
├── data/splits/folds.csv              partición fija por participante (test de 15 + 5 folds)
├── reports/
│   ├── informe_final.pdf / .docx      informe final (3 páginas)
│   ├── presentacion.pdf / .html       presentación (21 diapositivas)
│   ├── documento_tecnico.pdf / .docx  análisis completo complementario (11 páginas)
│   ├── *.csv                          predicciones y métricas guardadas que lee el cuaderno
│   └── figures/                       figuras del cuaderno y la presentación (incluye casos/ y estudio/)
├── material_estudio/                  guion, guía de estudio y apoyo para la sustentación (material interno)
├── empaquetar_entrega.py              arma el .zip de entrega
├── armar_repo_github.py               arma la carpeta que se sube a este repositorio
└── AGENTS.md                          contexto y estado del proyecto
```"""


def readme_repo():
    t = (ROOT / "README.md").read_text(encoding="utf-8")
    a, b = t.index("## Contenido"), t.index("## Qué no se incluye y por qué")
    contenido = ("## Estructura del repositorio\n\n" + TREE + "\n\n"
                 "El `.zip` que se envió al profesor (informe, presentación, documento técnico, cuaderno, `src/`, partición y CSV) "
                 "se arma con `python empaquetar_entrega.py --zip`. `material_estudio/` es de uso interno del grupo y no va en ese `.zip`.\n\n")
    t = t[:a] + contenido + t[b:]
    viejo = ("El cuaderno se entrega ejecutado, así que todas sus tablas y figuras se pueden leer sin volver a correrlo. "
             "Por tamaño, las figuras sueltas (`reports/figures/`) no se incluyen: van embebidas en el cuaderno y en la presentación, y los scripts de `src/` las regeneran.")
    assert viejo in t, "README.md cambió: revisar el reemplazo"
    t = t.replace(viejo, "El cuaderno está ejecutado, así que todas sus tablas y figuras se pueden leer sin volver a correrlo. "
                          "Las figuras sueltas están en `reports/figures/` y los scripts de `src/` las regeneran. "
                          "Los videos de demostración se comparten aparte, por el Drive del grupo, no en este repositorio.")
    return t


def copiar(src, rel):
    out = DEST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)


def main():
    assert DEST.parent == ENTREGA and DEST.name == "subir_a_github"
    if DEST.exists():
        shutil.rmtree(DEST)            # es una carpeta generada por este mismo script
    DEST.mkdir(parents=True)

    faltan = []
    for f in RAIZ:
        (copiar(ROOT / f, f) if (ROOT / f).exists() else faltan.append(f))
    for f in sorted((ROOT / "src").glob("*.py")):
        copiar(f, f"src/{f.name}")
    copiar(ROOT / "data" / "splits" / "folds.csv", "data/splits/folds.csv")
    for f in REPORTS:
        (copiar(ROOT / "reports" / f, f"reports/{f}") if (ROOT / "reports" / f).exists() else faltan.append("reports/" + f))
    for f in sorted((ROOT / "reports").glob("*.csv")):
        copiar(f, f"reports/{f.name}")
    for f in sorted((ROOT / "reports" / "figures").rglob("*.png")):
        copiar(f, f"reports/figures/{f.relative_to(ROOT / 'reports' / 'figures').as_posix()}")
    for origen, nombre in ESTUDIO.items():
        (copiar(ROOT / "reports" / origen, f"material_estudio/{nombre}") if (ROOT / "reports" / origen).exists() else faltan.append("reports/" + origen))
    (DEST / "README.md").write_text(readme_repo(), encoding="utf-8")
    if faltan:
        sys.exit("Faltan archivos: " + ", ".join(faltan))

    # ---- revisiones de seguridad
    problemas, rutas_abs = [], 0
    archivos = sorted(p for p in DEST.rglob("*") if p.is_file())
    for p in archivos:
        rel = p.relative_to(DEST).as_posix()
        if p.stat().st_size > MAX_MB * 1e6:
            problemas.append(f"{rel}: pesa más de {MAX_MB} MB")
        if p.suffix.lower() in PROHIBIDO_EXT:
            problemas.append(f"{rel}: extensión no permitida ({p.suffix})")
        if any(x in rel.lower() for x in PROHIBIDO_NOMBRE):
            problemas.append(f"{rel}: material de terceros")
        if p.suffix.lower() in TEXTOS:
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            if rel != "armar_repo_github.py":
                if CLAVE.search(p.read_text(encoding="utf-8", errors="ignore")):
                    problemas.append(f"{rel}: parece contener una clave")
                if p.suffix.lower() in CODIGO and "roboflow" in txt:
                    problemas.append(f"{rel}: menciona roboflow en el código")
                rutas_abs += txt.count("c:\\users\\lenovo") + txt.count("c:/users/lenovo")
    if problemas:
        shutil.rmtree(DEST)
        sys.exit("NO SE SUBE. Problemas encontrados:\n  " + "\n  ".join(problemas))

    total = sum(p.stat().st_size for p in archivos) / 1e6
    carpetas = {}
    for p in archivos:
        top = p.relative_to(DEST).parts[0] if len(p.relative_to(DEST).parts) > 1 else "(raíz)"
        carpetas.setdefault(top, [0, 0.0]); carpetas[top][0] += 1; carpetas[top][1] += p.stat().st_size / 1e6
    lineas = [f"Carpeta: entrega/subir_a_github  ({len(archivos)} archivos, {total:.1f} MB)", ""]
    lineas += [f"{n:4d} archivos  {mb:7.1f} MB  {k}" for k, (n, mb) in sorted(carpetas.items())]
    lineas += ["", "Revisiones: sin claves ni «roboflow» en el código, sin videos, pesos ni .zip, ningún archivo de más de "
                   f"{MAX_MB} MB, sin material de terceros."]
    if rutas_abs:
        lineas.append(f"Aviso: {rutas_abs} rutas absolutas de este computador aparecen en el texto (no es un secreto).")
    lineas += ["", "Archivos:"] + [f"  {p.relative_to(DEST).as_posix()}  ({p.stat().st_size / 1e6:.2f} MB)" for p in archivos]
    (ENTREGA / "MANIFIESTO_SUBIDA.txt").write_text("\n".join(lineas), encoding="utf-8")
    (ENTREGA / "INSTRUCCIONES_SUBIDA_GITHUB.md").write_text(instrucciones(len(archivos), total), encoding="utf-8")
    print("\n".join(lineas[:12]))
    print("\nListo. Instrucciones para quien haga el push: entrega/INSTRUCCIONES_SUBIDA_GITHUB.md")


def instrucciones(n, mb):
    return f"""# Instrucciones para subir el proyecto a GitHub

**Repositorio:** {REPO_URL} (rama `main`, privado)
**Qué se sube:** el contenido de la carpeta `entrega/subir_a_github/` ({n} archivos, {mb:.1f} MB). Nada más.

## Reglas

- Sube **solo** lo que está dentro de `entrega/subir_a_github/`. No agregues archivos de otras carpetas del proyecto.
- **No subas** datos ni videos (`visem-dataset/`, `VISEM-Tracking/`, `EVISAN/`, `data/processed/`, `reports/demo/`), pesos de
  modelos (`models/`, `weights/`, `runs/`, `*.pt`) ni la carpeta `reference-notebooks/` (trae una clave de API de otra persona).
- **No uses** `git push --force`. El repositorio remoto tiene un solo commit (`.gitignore`, `AGENTS.md` y el cuaderno viejo
  `00_proyecto_dl.ipynb`, que se reemplaza por `estimacion_motilidad_visem.ipynb`).
- No uses la subida por arrastre en la web de GitHub: admite como máximo 100 archivos y esta carpeta tiene {n}.
- Si algo falla (credenciales, rechazo del push), detente y avisa; no improvises.

## Pasos (PowerShell)

```powershell
# 1. Clonar el repositorio en una carpeta nueva (no toca el repositorio local del proyecto)
cd "C:\\Users\\Lenovo\\Desktop\\AA - Proyecto - RN\\entrega"
git clone {REPO_URL}.git clon_subida
cd clon_subida

# 2. Copiar encima el contenido de la carpeta de subida (incluye archivos ocultos como .gitignore)
Copy-Item -Recurse -Force "..\\subir_a_github\\*" .
Copy-Item -Force "..\\subir_a_github\\.gitignore" .

# 3. Quitar el cuaderno viejo, que ya fue reemplazado
git rm --ignore-unmatch 00_proyecto_dl.ipynb

# 4. Revisar antes de confirmar: deben ser ~{n} archivos y nada de datos, modelos ni videos
git add -A
git status --short
git diff --cached --stat | Select-Object -Last 3

# 5. Confirmar y subir
git commit -m "Proyecto final: estimación de motilidad espermática en VISEM (detección, seguimiento y Deep Learning)"
git push origin main
```

## Después de subir

1. Abrir {REPO_URL} y confirmar que se ven `README.md`, `src/`, `reports/`, `material_estudio/` y el cuaderno.
2. Confirmar que **no** aparecen `data/processed`, `models`, videos ni `reference-notebooks`.
3. Los 3 videos de demostración (`reports/demo/`, ~48 MB) se comparten aparte por el Drive del grupo.
"""


if __name__ == "__main__":
    main()
