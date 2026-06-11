"""Microservicio de detección de firma manuscrita en Notas de Encargo.

Arranque:
  .venv\\Scripts\\python.exe -m uvicorn servicio_firma:app --app-dir src --host 0.0.0.0 --port 8016

Variables de entorno opcionales:
  FIRMA_MODELO   ruta al .pt de YOLO (por defecto modelos/yolov8s-signature.pt)
  FIRMA_CONF     umbral de confianza (por defecto 0.25)
  FIRMA_PAGINAS  nº de últimas páginas a escanear (por defecto 2)
"""
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

import firma_yolo

_AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.environ.get(
    "FIRMA_MODELO", os.path.join(_AQUI, "..", "modelos", "yolov8s-signature.pt"))
CONF = float(os.environ.get("FIRMA_CONF", firma_yolo.CONF))
N_PAGINAS = int(os.environ.get("FIRMA_PAGINAS", firma_yolo.N_PAGINAS))

app = FastAPI(title="Detector de firmas - Nota de Encargo", version="1.0")

# El modelo se carga una sola vez al arrancar el proceso.
_modelo = YOLO(MODELO)


@app.get("/health")
def health():
    return {"status": "ok", "modelo": os.path.basename(MODELO),
            "conf": CONF, "n_paginas": N_PAGINAS}


@app.post("/detectar-firma")
async def detectar_firma(file: UploadFile = File(...)):
    nombre = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/octet-stream") \
            and not nombre.endswith(".pdf"):
        raise HTTPException(status_code=415,
                            detail=f"Solo se aceptan PDF. Recibido: {file.content_type or nombre}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío.")

    resultado = firma_yolo.detectar_firma(
        _modelo, pdf_bytes=data, conf=CONF, n_paginas=N_PAGINAS)

    if resultado.get("error"):
        raise HTTPException(status_code=422, detail=resultado["mensaje"])
    return resultado
