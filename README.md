# Detector de firmas — Nota de Encargo

Detecta si una Nota de Encargo (inmobiliaria) viene **firmada**, sea a mano o
digitalmente. Pendiente "VALIDAR FIRMA" del flujo n8n del CRM de Lion Capital.

## Cómo funciona (detección en capas)

1. **Firma digital (PAdES)** — con pyHanko se leen las firmas criptográficas
   embebidas (certificados FNMT/DNIe, plataformas). Es determinista y no renderiza:
   si hay firma digital, devuelve firmante(s), emisor, fecha e integridad.
2. **Firma manuscrita** (si no hay digital) — YOLO escanea las **últimas 2 páginas
   con contenido** (salta hojas en blanco) y marca firmado si encuentra una firma.
   Sobre la muestra actual: recall 13/13, verdadero negativo sobre el doc sin firmar.

## Estructura

```
src/
  firma_digital.py   # detección de firma digital PAdES (pyHanko)
  firma_yolo.py      # detección de firma manuscrita con YOLO
  servicio_firma.py  # microservicio FastAPI (orquesta las dos capas)
  detector_firma.py  # heurística inicial (descartada; aporta utilidades de página)
tests/
  test_firma_digital.py  # tests de caminos de error (firma digital)
  test_firma_yolo.py     # tests de caminos de error (YOLO)
  test_detector_firma.py # tests de la heurística
  test_yolo_firma.py     # banco de pruebas sobre las carpetas reales
modelos/
  yolov8s-signature.pt   # modelo preentrenado de firmas
```

## Despliegue con Docker (recomendado)

El modelo (`modelos/yolov8s-signature.pt`) ya viene incluido en el repo, así que el
despliegue es directo. El servicio escucha en el puerto **8016**.

```bash
git clone https://github.com/Gandrex87/detector_firmas.git detector_firmas
cd detector_firmas
docker compose up -d --build      # la 1ª build descarga torch CPU + ultralytics

docker ps | grep detector-firma   # debe verse (healthy) tras ~45s
curl http://localhost:8016/health
```

El `Dockerfile` usa `python:3.11-slim`, instala las librerías de sistema de OpenCV
(`libgl1`, `libglib2.0-0`), torch/torchvision **CPU-only** y pyHanko (Python puro).

## Puesta en marcha local (sin Docker)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn servicio_firma:app --app-dir src --host 0.0.0.0 --port 8016
```

> El modelo de firmas (`tech4humans/yolov8s-signature-detector`, AGPL) ya está en
> `modelos/`. Si hubiera que regenerarlo: está *gated* en HuggingFace
> (aceptar términos + login) y se descarga como `yolov8s.pt`.

Variables de entorno opcionales: `FIRMA_MODELO`, `FIRMA_CONF` (0.25), `FIRMA_PAGINAS` (2).

## Endpoints

- `GET /health` → estado y parámetros.
- `POST /detectar-firma` → multipart, campo `file` (PDF). El campo `tipo_firma`
  indica `"digital" | "manuscrita" | "ninguna"`.

Respuesta con **firma digital**:
```json
{
  "error": false,
  "firmado": true,
  "tipo_firma": "digital",
  "confianza": 1.0,
  "n_firmas_digitales": 2,
  "firmas_digitales": [
    {"campo": "Signature2", "firmante": "JAVIER GARCIA BOSQUET",
     "organizacion": "LION CAPITAL REAL ESTATE SL", "emisor": "FNMT-RCM / AC Representación",
     "fecha": "2026-06-10T13:30:26+02:00", "intacta": true}
  ]
}
```

Respuesta con **firma manuscrita**:
```json
{
  "error": false,
  "firmado": true,
  "tipo_firma": "manuscrita",
  "confianza": 0.747,
  "n_detecciones": 2,
  "paginas_escaneadas": [9, 10],
  "detecciones": [{"pagina": 9, "confianza": 0.747, "caja": [177.9, 1055.2, 558.9, 1258.1]}],
  "parametros": {"conf": 0.25, "n_paginas": 2, "dpi": 200}
}
```

Sin firma: `"firmado": false, "tipo_firma": "ninguna"`.
Códigos de error: `415` (no es PDF), `400` (vacío), `422` (PDF ilegible).

## Prueba rápida

```powershell
curl -X POST http://127.0.0.1:8016/detectar-firma -F "file=@ruta\documento.pdf;type=application/pdf"
```

O el banco de pruebas sobre las carpetas `firmada/` y `no firmada/`:

```powershell
.venv\Scripts\python.exe tests\test_yolo_firma.py
```

## Integración en n8n

El flujo `Validador NOTA_DE_ENCARGO - CATASTRO` lo llama por la IP del host:
`POST http://10.1.0.188:8016/detectar-firma` (campo binario `file`).

Durante pruebas corre en modo *shadow*: se conecta **después** del
`Respond - NOTA DE ENCARGO` (la respuesta a producción ya salió, impacto cero) y
registra cada resultado en la tabla Postgres `pruebas_firma`.
