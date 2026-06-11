"""Detección de firma digital (PAdES) en PDFs con pyHanko.

A diferencia de la firma manuscrita (que se 've' y se detecta con YOLO), la firma
digital vive en la estructura criptográfica del PDF. Esta detección es determinista:
lee los objetos de firma embebidos, identifica al firmante y comprueba integridad.
"""
import io
from pyhanko.pdf_utils.reader import PdfFileReader
from pyhanko.sign.validation import validate_pdf_signature
from pyhanko_certvalidator import ValidationContext


def _nombre_firmante(cert):
    sub = cert.subject.native
    given, surname = sub.get("given_name"), sub.get("surname")
    if given and surname:
        return f"{given} {surname}".strip()
    return sub.get("common_name") or "(desconocido)"


def _emisor(cert):
    iss = cert.issuer.native
    partes = [iss.get("organization_name"), iss.get("common_name")]
    return " / ".join(x for x in partes if x) or "(desconocido)"


def detectar_firma_digital(pdf_path=None, pdf_bytes=None, validar=True):
    """Detecta firmas digitales PAdES embebidas en el PDF.
    Pasa `pdf_path` o `pdf_bytes`. Nunca lanza ante un PDF inválido."""
    if pdf_bytes is None and pdf_path is None:
        return {"error": True, "mensaje": "Falta pdf_path o pdf_bytes."}

    stream = None
    try:
        stream = io.BytesIO(pdf_bytes) if pdf_bytes is not None else open(pdf_path, "rb")
        r = PdfFileReader(stream)
        sigs = list(r.embedded_signatures)
        vc = ValidationContext(allow_fetching=False) if validar else None

        firmas = []
        for sig in sigs:
            cert = sig.signer_cert
            info = {
                "campo": sig.field_name,
                "firmante": _nombre_firmante(cert),
                "organizacion": cert.subject.native.get("organization_name"),
                "emisor": _emisor(cert),
                "fecha": sig.self_reported_timestamp.isoformat()
                if sig.self_reported_timestamp else None,
            }
            if validar:
                try:
                    st = validate_pdf_signature(sig, vc)
                    info["intacta"] = bool(st.intact and st.valid)
                except Exception as e:
                    info["intacta"] = None
                    info["validacion_error"] = str(e)[:150]
            firmas.append(info)

        return {
            "error": False,
            "tiene_firma_digital": len(firmas) > 0,
            "n_firmas": len(firmas),
            "firmas": firmas,
        }
    except Exception as e:
        return {"error": True, "mensaje": f"Error leyendo firmas digitales: {e}"}
    finally:
        if stream is not None:
            stream.close()
