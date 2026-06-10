# tests/test_fiabilidad.py
"""Corre el detector sobre las carpetas reales y reporta fiabilidad.
Uso: .venv/Scripts/python.exe tests/test_fiabilidad.py
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fitz
import detector_firma as df

BASE = r"C:\Users\andres\Documents\Nota encargo - catastro"
FIRMADA = os.path.join(BASE, "firmada")
NO_FIRMADA = os.path.join(BASE, "no firmada")
DEBUG = os.path.join(os.path.dirname(__file__), "debug")
SINTET = os.path.join(os.path.dirname(__file__), "sinteticos_no_firmada")


def generar_negativos_sinteticos():
    """Crea versiones 'no firmadas' pintando en blanco la región de firma
    de cada PDF firmado (región geométrica fija, no el trazo detectado)."""
    os.makedirs(SINTET, exist_ok=True)
    rutas = []
    for p in sorted(glob.glob(os.path.join(FIRMADA, "*.pdf"))):
        try:
            doc = fitz.open(p)
            page = doc[df._indice_pagina_con_firma(doc)]  # misma página que analiza el detector
            r = page.rect
            fx0, fy0, fx1, fy1 = df.REGION
            zona = fitz.Rect(r.width*fx0, r.height*fy0, r.width*fx1, r.height*fy1)
            page.add_redact_annot(zona, fill=(1, 1, 1))
            page.apply_redactions()  # blanquea los píxeles de la zona (incluye escaneos de página completa)
            destino = os.path.join(SINTET, os.path.basename(p))
            doc.save(destino)
            doc.close()
            rutas.append(destino)
        except Exception as e:
            print("  (no se pudo sintetizar", os.path.basename(p), ":", e, ")")
    return rutas


def correr(rutas, etiqueta_esperada):
    print(f"\n=== {etiqueta_esperada.upper()} ({len(rutas)} docs) ===")
    aciertos = 0
    for p in rutas:
        r = df.detectar(p, debug_dir=DEBUG)
        if r["error"]:
            print(f"  ERROR {os.path.basename(p)[:45]:45} -> {r['mensaje']}")
            continue
        ok = (r["firmado"] == (etiqueta_esperada == "firmada"))
        aciertos += int(ok)
        marca = "OK " if ok else "XX "
        d = r["detalles"]
        print(f"  {marca}{os.path.basename(p)[:42]:42} firmado={str(r['firmado']):5} "
              f"conf={r['confianza']:.2f} ink={d['ink_ratio']:.4f} trazos={d['n_trazos']} "
              f"metodo={r['metodo']}")
    print(f"  -> aciertos: {aciertos}/{len(rutas)}")
    return aciertos, len(rutas)


def main():
    firmadas = sorted(glob.glob(os.path.join(FIRMADA, "*.pdf")))
    no_firmadas = sorted(glob.glob(os.path.join(NO_FIRMADA, "*.pdf")))
    print(f"REGION={df.REGION}  UMBRAL_INK={df.UMBRAL_INK}  DPI={df.DPI}")
    correr(firmadas, "firmada")
    if no_firmadas:
        correr(no_firmadas, "no firmada")
    else:
        print("\n(no hay 'no firmada' reales; usando negativos sintéticos)")
        sint = generar_negativos_sinteticos()
        correr(sint, "no firmada")
    print(f"\nRecortes de debug en: {DEBUG}")


if __name__ == "__main__":
    main()
