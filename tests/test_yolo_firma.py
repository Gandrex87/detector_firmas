"""Prueba del modelo YOLO preentrenado de detección de firmas sobre las
Notas de Encargo. Lee la PÁGINA COMPLETA (última con contenido) — el recorte
inferior-izquierda quedó descartado (1/9 vs 8/9 en página completa).

Mide:
  - Recall sobre 'firmada'      (¿detecta la firma cuando la hay?)
  - Especificidad sobre 'no firmada' (¿acierta diciendo 'sin firma'?)

Uso:
  .venv\\Scripts\\python.exe tests\\test_yolo_firma.py            # conf=0.25
  .venv\\Scripts\\python.exe tests\\test_yolo_firma.py 0.15       # otro umbral
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


def pagina_rgb(pdf_path):
    """Render RGB de la última página con contenido (ignora hojas en blanco)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[df._indice_pagina_con_firma(doc)]
        pix = page.get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img.copy()
    finally:
        doc.close()


def detectar(model, pdf_path, etiqueta_carpeta):
    """Corre YOLO sobre la página completa; devuelve (n, max_conf, cajas)."""
    img = pagina_rgb(pdf_path)
    res = model.predict(img, conf=CONF, verbose=False)[0]
    cajas = []
    h, w = img.shape[:2]
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cajas.append({"conf": round(float(b.conf[0]), 3),
                      "cx": round((x1 + x2) / 2 / w, 2),
                      "cy": round((y1 + y2) / 2 / h, 2)})
    os.makedirs(DEBUG, exist_ok=True)
    nombre = os.path.splitext(os.path.basename(pdf_path))[0][:40]
    cv2.imwrite(os.path.join(DEBUG, f"[{etiqueta_carpeta}] {nombre}.png"), res.plot())
    maxc = max((k["conf"] for k in cajas), default=0.0)
    return len(cajas), maxc, cajas


def correr(model, carpeta, etiqueta, espera_firma):
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.pdf")))
    print(f"\n=== {etiqueta.upper()} ({len(rutas)} docs)  [esperado: {'CON firma' if espera_firma else 'SIN firma'}] ===")
    aciertos = 0
    for p in rutas:
        n, maxc, cajas = detectar(model, p, etiqueta)
        firmado = n > 0
        ok = (firmado == espera_firma)
        aciertos += int(ok)
        pos = f" en {[(c['cx'], c['cy']) for c in cajas]}" if cajas else ""
        print(f"  {'OK ' if ok else 'XX '}{os.path.basename(p)[:44]:44} "
              f"{'FIRMADO' if firmado else 'sin firma':9} ({n} det, max={maxc:.2f}){pos}")
    print(f"  -> aciertos: {aciertos}/{len(rutas)}")
    return aciertos, len(rutas)


def main():
    if not os.path.exists(MODELO):
        print(f"ERROR: falta el modelo en {MODELO}")
        sys.exit(1)
    from ultralytics import YOLO
    model = YOLO(MODELO)
    print(f"Modelo: yolov8s-signature  |  conf>={CONF}  |  DPI={DPI}  |  página completa")
    correr(model, FIRMADA, "firmada", espera_firma=True)
    if glob.glob(os.path.join(NO_FIRMADA, "*.pdf")):
        correr(model, NO_FIRMADA, "no firmada", espera_firma=False)
    else:
        print("\n(carpeta 'no firmada' vacía)")
    print(f"\nImágenes anotadas en: {DEBUG}")


if __name__ == "__main__":
    main()
