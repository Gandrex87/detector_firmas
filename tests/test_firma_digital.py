"""Tests de los caminos de error de firma_digital.detectar_firma_digital."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import firma_digital as fd


def test_sin_entrada_devuelve_error():
    r = fd.detectar_firma_digital()
    assert r["error"] is True
    assert "mensaje" in r


def test_bytes_invalidos_devuelve_error():
    r = fd.detectar_firma_digital(pdf_bytes=b"%PDF-1.4 no es un pdf valido")
    assert r["error"] is True
    assert "mensaje" in r
