# tests/test_detector_firma.py
import numpy as np
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
