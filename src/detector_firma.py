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
