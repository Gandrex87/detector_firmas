"""Prueba del modelo YOLO preentrenado de detección de firmas sobre las
Notas de Encargo firmadas. Compara dos modos:
  (A) página completa
  (B) recorte inferior-izquierda (la zona donde 'debería' ir la firma)

Uso:
  .venv\\Scripts\\python.exe tests\\test_yolo_firma.py
  .venv\\Scripts\\python.exe tests\\test_yolo_firma.py 0.15   # umbral de confianza

Requiere el modelo en modelos/yolov8s-signature.pt (ver instrucciones de descarga).
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fitz
import cv2
import numpy as np
import detector_firma as df

BASE = r"C:\Users\andres\Documents\Nota encargo - catastro"
FIRMADA = os.path.join(BASE, "firmada")
MODELO = os.path.join(os.path.dirname(__file__), "..", "modelos", "yolov8s-signature.pt")
DEBUG = os.path.join(os.path.dirname(__file__), "debug_yolo")
DPI = 200
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25


def pagina_rgb(pdf_path):
    """Render RGB de la última página con contenido + su rect."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[df._indice_pagina_con_firma(doc)]
        pix = page.get_pixmap(dpi=DPI)  # RGB
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img.copy()
    finally:
        doc.close()


def detectar_en(model, img, etiqueta, nombre):
    """Corre YOLO sobre una imagen RGB; devuelve (n_detecciones, max_conf, lista_cajas)."""
    res = model.predict(img, conf=CONF, verbose=False)[0]
    cajas = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        c = float(b.conf[0])
        h, w = img.shape[:2]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        cajas.append({"conf": round(c, 3), "cx": round(cx, 2), "cy": round(cy, 2)})
    # guardar imagen anotada
    os.makedirs(DEBUG, exist_ok=True)
    anot = res.plot()
    cv2.imwrite(os.path.join(DEBUG, f"{nombre}__{etiqueta}.png"), anot)
    n = len(cajas)
    maxc = max((k["conf"] for k in cajas), default=0.0)
    return n, maxc, cajas


def main():
    if not os.path.exists(MODELO):
        print(f"ERROR: no se encuentra el modelo en {MODELO}")
        print("Descárgalo primero (ver instrucciones).")
        sys.exit(1)
    from ultralytics import YOLO
    model = YOLO(MODELO)
    print(f"Modelo: {MODELO}  |  conf>={CONF}  |  DPI={DPI}\n")
    print(f"{'documento':42} | {'PAGINA COMPLETA':28} | {'RECORTE INF-IZQ':20}")
    print("-" * 96)
    ok_full = ok_crop = total = 0
    for p in sorted(glob.glob(os.path.join(FIRMADA, "*.pdf"))):
        nombre = os.path.splitext(os.path.basename(p))[0][:40]
        img = pagina_rgb(p)
        # (A) página completa
        nf, cf, cajasf = detectar_en(model, img, "full", nombre)
        # (B) recorte inferior-izquierda
        sub, _bbox = df.recortar_region(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), df.REGION)
        h, w = img.shape[:2]
        fx0, fy0, fx1, fy1 = df.REGION
        crop = img[int(h*fy0):int(h*fy1), int(w*fx0):int(w*fx1)]
        nc, cc, cajasc = detectar_en(model, crop, "crop", nombre)

        total += 1
        ok_full += int(nf > 0)
        ok_crop += int(nc > 0)
        det_f = f"{nf} det, max={cf:.2f} cx,cy={cajasf[0]['cx'],cajasf[0]['cy']}" if nf else "0 det"
        det_c = f"{nc} det, max={cc:.2f}" if nc else "0 det"
        print(f"{nombre:42} | {det_f:28} | {det_c:20}")
    print("-" * 96)
    print(f"Firmadas detectadas -> pagina completa: {ok_full}/{total}   recorte inf-izq: {ok_crop}/{total}")
    print(f"\nImágenes anotadas en: {DEBUG}")


if __name__ == "__main__":
    main()
