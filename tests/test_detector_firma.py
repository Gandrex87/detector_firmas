# tests/test_detector_firma.py
import numpy as np
import cv2
import fitz
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import detector_firma as df


def _pdf_dos_paginas(tmp_path):
    """Crea un PDF de 2 páginas; la 2ª (última) con un rectángulo negro abajo-izquierda."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)            # página 1 en blanco
    page = doc.new_page(width=595, height=842)      # página 2 (última)
    page.draw_rect(fitz.Rect(30, 700, 200, 800), color=(0, 0, 0), fill=(0, 0, 0))
    ruta = str(tmp_path / "doc.pdf")
    doc.save(ruta)
    doc.close()
    return ruta


def test_render_ultima_pagina_devuelve_gris(tmp_path):
    ruta = _pdf_dos_paginas(tmp_path)
    img, rect = df.render_ultima_pagina(ruta, dpi=150)
    assert img.ndim == 2                    # escala de grises
    assert img.dtype == np.uint8
    assert img.shape[0] > 0 and img.shape[1] > 0
    assert rect.width == 595 and rect.height == 842


def test_recortar_region_inferior_izquierda():
    img = np.full((1000, 800), 255, dtype=np.uint8)  # alto=1000, ancho=800
    sub, bbox = df.recortar_region(img, (0.0, 0.60, 0.55, 1.0))
    x0, y0, x1, y1 = bbox
    assert (x0, y0) == (0, 600)
    assert (x1, y1) == (440, 1000)
    assert sub.shape == (400, 440)


def test_score_tinta_region_en_blanco():
    blanco = np.full((400, 440), 255, dtype=np.uint8)
    r = df.score_tinta(blanco)
    assert r["ink_ratio"] < 0.001
    assert r["n_trazos"] == 0


def test_score_tinta_con_trazo():
    img = np.full((400, 440), 255, dtype=np.uint8)
    # simula una firma: varias curvas/líneas negras
    cv2.line(img, (40, 300), (380, 320), 0, 4)
    cv2.line(img, (60, 340), (300, 280), 0, 3)
    cv2.ellipse(img, (200, 310), (90, 40), 0, 0, 300, 0, 3)
    r = df.score_tinta(img)
    assert r["ink_ratio"] > 0.005
    assert r["n_trazos"] >= 1


def test_imagen_en_region_falsa_sin_imagenes(tmp_path):
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "Texto sin imagenes")
    ruta = str(tmp_path / "sintexto.pdf")
    doc.save(ruta); doc.close()

    doc2 = fitz.open(ruta)
    page2 = doc2[-1]
    assert df.imagen_en_region(page2, df.REGION) is False
    doc2.close()
