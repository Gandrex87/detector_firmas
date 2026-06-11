# Detector de firmas — Nota de Encargo

Detecta si una Nota de Encargo (inmobiliaria) viene **firmada a mano**, usando un
modelo YOLO de detección de firmas. Fase 1 del pendiente "VALIDAR FIRMA" del flujo
n8n del CRM de Lion Capital.

## Cómo funciona

Escanea las **últimas 2 páginas con contenido** del PDF (salta hojas en blanco
residuales del escáner) y marca `firmado` si YOLO encuentra una firma en alguna.
Resultados sobre la muestra actual: recall 13/13, verdadero negativo sobre el doc
sin firmar.

## Estructura

```
src/
  firma_yolo.py      # lógica de detección con YOLO (sin HTTP)
  servicio_firma.py  # microservicio FastAPI
  detector_firma.py  # heurística inicial (descartada como principal; aporta utilidades de página)
tests/
  test_firma_yolo.py     # tests de caminos de error
  test_detector_firma.py # tests de la heurística
  test_yolo_firma.py     # banco de pruebas sobre las carpetas reales
modelos/
  yolov8s-signature.pt   # modelo preentrenado (no versionado; ver descarga)
```

## Puesta en marcha

```powershell
# 1. Entorno
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Modelo (gated en HuggingFace: aceptar términos + login una vez)
#    https://huggingface.co/tech4humans/yolov8s-signature-detector
#    luego descargar yolov8s.pt a modelos/yolov8s-signature.pt

# 3. Arrancar el microservicio
.venv\Scripts\python.exe -m uvicorn servicio_firma:app --app-dir src --host 0.0.0.0 --port 8030
```

Variables de entorno opcionales: `FIRMA_MODELO`, `FIRMA_CONF` (0.25), `FIRMA_PAGINAS` (2).

## Endpoints

- `GET /health` → estado y parámetros.
- `POST /detectar-firma` → multipart, campo `file` (PDF). Respuesta:

```json
{
  "error": false,
  "firmado": true,
  "confianza": 0.747,
  "n_detecciones": 2,
  "paginas_escaneadas": [9, 10],
  "detecciones": [{"pagina": 9, "confianza": 0.747, "caja": [177.9, 1055.2, 558.9, 1258.1]}],
  "parametros": {"conf": 0.25, "n_paginas": 2, "dpi": 200}
}
```

Códigos de error: `415` (no es PDF), `400` (vacío), `422` (PDF ilegible).

## Prueba rápida

```powershell
curl -X POST http://127.0.0.1:8030/detectar-firma -F "file=@ruta\documento.pdf;type=application/pdf"
```

O el banco de pruebas sobre las carpetas `firmada/` y `no firmada/`:

```powershell
.venv\Scripts\python.exe tests\test_yolo_firma.py
```
