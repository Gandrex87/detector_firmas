# src/detector_firma.py
"""Detección de firma manuscrita en Notas de Encargo (Fase 1)."""
import fitz
import cv2
import numpy as np

# --- Configuración ajustable ---
DPI = 200
# Región de firma como fracciones del rect de la página: (x0, y0, x1, y1)
REGION = (0.00, 0.60, 0.55, 1.00)
UMBRAL_INK = 0.010   # ratio de píxeles de tinta para considerar "firmado" (se afina con el banco de pruebas)
AREA_MIN_TRAZO = 40  # área mínima (px) de un componente para contar como trazo


def render_ultima_pagina(pdf_path, dpi=DPI):
    """Devuelve (imagen_gris_uint8, rect_pagina) de la última página."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[-1]
        rect = page.rect
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return img.copy(), rect
    finally:
        doc.close()


def recortar_region(img, region=REGION):
    """Recorta una región (fracciones x0,y0,x1,y1) de una imagen gris.
    Devuelve (subimagen, (px0, py0, px1, py1))."""
    h, w = img.shape[:2]
    fx0, fy0, fx1, fy1 = region
    x0, y0 = int(w * fx0), int(h * fy0)
    x1, y1 = int(w * fx1), int(h * fy1)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def score_tinta(region_img):
    """Mide presencia de tinta manuscrita en una región gris.
    Devuelve dict con ink_ratio, n_trazos."""
    # Binariza: tinta (oscuro) -> 255
    binaria = cv2.adaptiveThreshold(
        region_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 15,
    )
    total = region_img.shape[0] * region_img.shape[1]
    ink = int(np.count_nonzero(binaria))
    ink_ratio = ink / total if total else 0.0

    # Componentes conectados: cuenta "trazos" significativos
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    n_trazos = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= AREA_MIN_TRAZO))

    return {"ink_ratio": round(ink_ratio, 5), "n_trazos": n_trazos}


def imagen_en_region(page, region=REGION):
    """True si hay una imagen raster colocada dentro de la región de firma,
    y la página NO es un escaneo de página completa (tiene texto).
    En escaneos (sin texto) esta señal no es fiable y devuelve False."""
    texto = page.get_text().strip()
    if not texto:
        return False  # escaneo: toda la página es imagen; señal no aplicable
    rect = page.rect
    fx0, fy0, fx1, fy1 = region
    zona = fitz.Rect(rect.width * fx0, rect.height * fy0,
                     rect.width * fx1, rect.height * fy1)
    for img in page.get_image_info():
        bbox = fitz.Rect(img["bbox"])
        if zona.intersects(bbox):
            return True
    return False
