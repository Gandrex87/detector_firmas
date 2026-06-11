"""Prueba del modelo YOLO preentrenado de detección de firmas sobre las
Notas de Encargo. Lee la PÁGINA COMPLETA (el recorte quedó descartado).

Escanea las últimas N páginas CON CONTENIDO (salta hojas en blanco) y marca
'firmado' si alguna de ellas tiene una firma. Por defecto N=2.

Mide:
  - Recall sobre 'firmada'            (¿detecta la firma cuando la hay?)
  - Especificidad sobre 'no firmada'  (¿acierta diciendo 'sin firma'?)

Uso:
  .venv\\Scripts\\python.exe tests\\test_yolo_firma.py             # conf=0.25, 2 páginas
  .venv\\Scripts\\python.exe tests\\test_yolo_firma.py 0.25 2      # conf, nº de páginas
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fitz
import cv2
import numpy as np
import detector_firma as df

BASE = r"C:\Users\andres\Documents\Nota encargo - catastro"
FIRMADA = os.path.join(BASE, "firmada")
NO_FIRMADA = os.path.join(BASE, "no firmada")
MODELO = os.path.join(os.path.dirname(__file__), "..", "modelos", "yolov8s-signature.pt")
DEBUG = os.path.join(os.path.dirname(__file__), "debug_yolo")
DPI = 200
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
N_PAGINAS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def indices_a_escanear(doc, n):
    """Últimos n índices de página CON contenido (salta hojas en blanco), ordenados."""
    idxs = []
    for i in range(doc.page_count - 1, -1, -1):
        pix = doc[i].get_pixmap(dpi=72, colorspace=fitz.csGRAY)
        g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        if df._ink_ratio(g) > df.PAGINA_BLANCA_MAX_INK:
            idxs.append(i)
            if len(idxs) >= n:
                break
    return sorted(idxs) if idxs else [doc.page_count - 1]


def render_rgb(page):
    pix = page.get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img.copy()


def detectar(model, pdf_path, etiqueta):
    """Escanea las últimas N páginas con contenido. Devuelve (firmado, detalle)."""
    doc = fitz.open(pdf_path)
    try:
        idxs = indices_a_escanear(doc, N_PAGINAS)
        nombre = os.path.splitext(os.path.basename(pdf_path))[0][:38]
        hits = []   # (pagina, max_conf)
        for i in idxs:
            img = render_rgb(doc[i])
            res = model.predict(img, conf=CONF, verbose=False)[0]
            os.makedirs(DEBUG, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG, f"[{etiqueta}] {nombre} p{i+1}.png"), res.plot())
            if len(res.boxes):
                hits.append((i + 1, round(float(max(b.conf[0] for b in res.boxes)), 2)))
        firmado = len(hits) > 0
        detalle = f"págs {[i+1 for i in idxs]} -> " + (
            f"firma en {hits}" if hits else "sin detección")
        return firmado, detalle
    finally:
        doc.close()


def correr(model, carpeta, etiqueta, espera_firma):
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.pdf")))
    print(f"\n=== {etiqueta.upper()} ({len(rutas)} docs)  [esperado: {'CON firma' if espera_firma else 'SIN firma'}] ===")
    aciertos = 0
    for p in rutas:
        firmado, detalle = detectar(model, p, etiqueta)
        ok = (firmado == espera_firma)
        aciertos += int(ok)
        print(f"  {'OK ' if ok else 'XX '}{os.path.basename(p)[:42]:42} "
              f"{'FIRMADO' if firmado else 'sin firma':9} | {detalle}")
    print(f"  -> aciertos: {aciertos}/{len(rutas)}")
    return aciertos, len(rutas)


def main():
    if not os.path.exists(MODELO):
        print(f"ERROR: falta el modelo en {MODELO}")
        sys.exit(1)
    from ultralytics import YOLO
    model = YOLO(MODELO)
    print(f"Modelo: yolov8s-signature  |  conf>={CONF}  |  DPI={DPI}  |  últimas {N_PAGINAS} págs con contenido")
    correr(model, FIRMADA, "firmada", espera_firma=True)
    if glob.glob(os.path.join(NO_FIRMADA, "*.pdf")):
        correr(model, NO_FIRMADA, "no firmada", espera_firma=False)
    else:
        print("\n(carpeta 'no firmada' vacía)")
    print(f"\nImágenes anotadas en: {DEBUG}")


if __name__ == "__main__":
    main()
