"""Tests de los caminos de error de firma_yolo.detectar_firma.
No requieren el modelo YOLO (fallan antes de usarlo)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import firma_yolo


def test_sin_entrada_devuelve_error():
    r = firma_yolo.detectar_firma(model=None)
    assert r["error"] is True
    assert "mensaje" in r


def test_bytes_invalidos_devuelve_error():
    # Un PDF corrupto no debe lanzar excepción, sino devolver error controlado.
    r = firma_yolo.detectar_firma(model=None, pdf_bytes=b"%PDF-1.4 basura no valida")
    assert r["error"] is True
    assert "mensaje" in r
