FROM python:3.11-slim

WORKDIR /app

# Dependencias de sistema que OpenCV y PyMuPDF necesitan en la imagen slim.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# Torch + torchvision CPU-only desde el índice oficial de PyTorch: evita ~2 GB de
# CUDA y, sobre todo, garantiza que torch y torchvision sean versiones COMPATIBLES
# (si torchvision se instala desde PyPI por defecto, falla con
# "operator torchvision::nms does not exist"). Luego el resto de dependencias.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-docker.txt

# Código y modelo
COPY src/ ./src/
COPY modelos/yolov8s-signature.pt ./modelos/yolov8s-signature.pt

ENV FIRMA_MODELO=/app/modelos/yolov8s-signature.pt \
    FIRMA_CONF=0.25 \
    FIRMA_PAGINAS=2

EXPOSE 8016

# El modelo tarda unos segundos en cargar al arrancar -> start-period holgado.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8016/health')" || exit 1

CMD ["uvicorn", "servicio_firma:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8016"]
