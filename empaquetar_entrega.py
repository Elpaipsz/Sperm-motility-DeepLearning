"""Arma la carpeta de entrega (y, con --zip, el .zip) con exactamente lo que describe README.md.

Uso:
    python empaquetar_entrega.py          # crea o refresca entrega/ProyectoFinal_DL_MotilidadVISEM/
    python empaquetar_entrega.py --zip    # además crea entrega/ProyectoFinal_DL_MotilidadVISEM.zip

No incluye datos originales ni procesados, modelos, videos ni material interno (AGENTS.md, guion,
cuadernos de referencia de terceros). Antes de copiar revisa que el cuaderno esté ejecutado y sin errores.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "ProyectoFinal_DL_MotilidadVISEM"
DEST = ROOT / "entrega" / NAME
NOTEBOOK = "estimacion_motilidad_visem.ipynb"
GMAIL_LIMIT_MB = 25


def contenido():
    files = [ROOT / "README.md", ROOT / NOTEBOOK, ROOT / "requirements.txt", ROOT / "data" / "splits" / "folds.csv"]
    files += sorted((ROOT / "src").glob("*.py"))
    files += sorted((ROOT / "reports").glob("*.csv"))
    # las figuras van embebidas en el cuaderno ejecutado y en la presentación; src/ las regenera (se omiten por tamaño)
    return files


def revisar_cuaderno():
    nb = json.loads((ROOT / NOTEBOOK).read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ejecutadas = sum(1 for c in code if c.get("execution_count"))
    errores = [o.get("ename") for c in code for o in c.get("outputs", []) if o["output_type"] == "error"]
    print(f"Cuaderno: {len(nb['cells'])} celdas, {ejecutadas}/{len(code)} celdas de código ejecutadas, {len(errores)} errores")
    if ejecutadas < len(code) or errores:
        print("  AVISO: el cuaderno no está completamente ejecutado o tiene errores. Ejecútelo (Run All) antes de entregar.")
    return ejecutadas == len(code) and not errores


def main():
    ok = revisar_cuaderno()
    files = contenido()
    faltan = [f for f in files if not f.exists()]
    if faltan:
        sys.exit("Faltan archivos: " + ", ".join(str(f.relative_to(ROOT)) for f in faltan))
    # mapa destino-dentro-del-paquete -> archivo original
    plan = {f.relative_to(ROOT).as_posix(): f for f in files}
    # los dos documentos que se evalúan van en la raíz del paquete, a la vista al abrir el .zip
    for f in ("informe_final.pdf", "presentacion.pdf", "presentacion.html"):
        assert (ROOT / "reports" / f).exists(), f
        plan[f] = ROOT / "reports" / f
    # documento técnico: material complementario (el informe evaluado sigue siendo informe_final.pdf, 3 páginas)
    plan["documento_tecnico_complementario.pdf"] = ROOT / "reports" / "documento_tecnico.pdf"

    # carpeta de revisión: se refresca archivo por archivo; si uno está abierto en otro programa, se avisa y se sigue
    assert DEST.parent == ROOT / "entrega" and DEST.name == NAME
    bloqueados = []
    for rel, src in plan.items():
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, out)
        except PermissionError:
            bloqueados.append(rel)
    for f in sorted(DEST.rglob("*"), reverse=True):   # quita lo que ya no forma parte del paquete
        rel = f.relative_to(DEST).as_posix()
        if f.is_file() and rel not in plan:
            try:
                f.unlink()
            except PermissionError:
                bloqueados.append(rel)
        elif f.is_dir() and not any(f.iterdir()):
            f.rmdir()
    total = sum(src.stat().st_size for src in plan.values()) / 1e6
    print(f"Carpeta: {DEST.relative_to(ROOT)}  ({len(plan)} archivos, {total:.1f} MB)")
    if bloqueados:
        print("  AVISO: no se pudieron actualizar en la carpeta (están abiertos en otro programa):", ", ".join(bloqueados))
    if "--zip" in sys.argv:
        if not ok:
            print("  El .zip se crea igual, pero revise el aviso del cuaderno.")
        z = ROOT / "entrega" / f"{NAME}.zip"
        # el .zip se arma desde los archivos originales: un archivo abierto en la carpeta de revisión no lo afecta
        with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for rel, src in sorted(plan.items()):
                zf.write(src, f"{NAME}/{rel}")
        mb = z.stat().st_size / 1e6
        print(f"Zip: {z.relative_to(ROOT)}  ({mb:.1f} MB)" + ("" if mb < GMAIL_LIMIT_MB else f"  AVISO: supera {GMAIL_LIMIT_MB} MB, el límite de adjuntos de Gmail"))


if __name__ == "__main__":
    main()
