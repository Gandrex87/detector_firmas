"""Detección de firma manuscrita en Notas de Encargo con YOLO.

Escanea las últimas N páginas con contenido (salta hojas en blanco) y marca
'firmado' si alguna tiene una firma. Lógica pura, sin HTTP: recibe el modelo
ya cargado para no recargarlo en cada llamada.
"""
import fitz
import cv2
import numpy as np
from detector_firma import _ink_ratio, PAGINA_BLANCA_MAX_INK

DPI = 200
CONF = 0.25
N_PAGINAS = 2


def _indices_a_escanear(doc, n):
    """Últimos n índices de página CON contenido (salta hojas en blanco), ordenados."""
    idxs = []
    for i in range(doc.page_count - 1, -1, -1):
        pix = doc[i].get_pixmap(dpi=72, colorspace=fitz.csGRAY)
        g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        if _ink_ratio(g) > PAGINA_BLANCA_MAX_INK:
            idxs.append(i)
            if len(idxs) >= n:
                break
    return sorted(idxs) if idxs else [doc.page_count - 1]


def _render_rgb(page, dpi):
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img.copy()


def detectar_firma(model, pdf_path=None, pdf_bytes=None,
                   conf=CONF, n_paginas=N_PAGINAS, dpi=DPI):
    """Detecta firma manuscrita en un PDF. Pasa `pdf_path` o `pdf_bytes`.
    `model` es un YOLO ya cargado. Devuelve un dict estructurado; ante un PDF
    inválido devuelve {'error': True, 'mensaje': ...} en lugar de lanzar."""
    if pdf_bytes is None and pdf_path is None:
        return {"error": True, "mensaje": "Falta pdf_path o pdf_bytes."}

    doc = None
    try:
        if pdf_bytes is not None:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return {"error": True, "mensaje": "El PDF no tiene páginas."}
        idxs = _indices_a_escanear(doc, n_paginas)
        detecciones = []
        for i in idxs:
            img = _render_rgb(doc[i], dpi)
            res = model.predict(img, conf=conf, verbose=False)[0]
            for b in res.boxes:
                detecciones.append({
                    "pagina": i + 1,
                    "confianza": round(float(b.conf[0]), 3),
                    "caja": [round(v, 1) for v in b.xyxy[0].tolist()],
                })
        confianza = max((d["confianza"] for d in detecciones), default=0.0)
        return {
            "error": False,
            "firmado": len(detecciones) > 0,
            "confianza": confianza,
            "n_detecciones": len(detecciones),
            "paginas_escaneadas": [i + 1 for i in idxs],
            "detecciones": detecciones,
            "parametros": {"conf": conf, "n_paginas": n_paginas, "dpi": dpi},
        }
    except Exception as e:
        return {"error": True, "mensaje": f"Error procesando PDF: {e}"}
    finally:
        if doc is not None:
            doc.close()
