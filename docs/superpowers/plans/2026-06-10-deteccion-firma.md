# Detección de firma manuscrita — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detectar si una Nota de Encargo trae firma manuscrita en la última página (zona inferior izquierda) y medir la fiabilidad sobre las muestras reales.

**Architecture:** Núcleo Python que renderiza la última página del PDF con PyMuPDF, recorta la región de firma y puntúa la presencia de tinta manuscrita con OpenCV. Un banco de pruebas corre el núcleo sobre las carpetas `firmada/` y `no firmada/`, genera negativos sintéticos y reporta métricas.

**Tech Stack:** Python 3.13, PyMuPDF (fitz), OpenCV, NumPy. Entorno en `.venv`.

---

## Notas de contexto (de la inspección de muestras)

- 8/9 muestras son **escaneos** (última página = una imagen de página completa, sin texto). 1 es PDF nativo con texto.
- Por tanto la **señal principal** es densidad/forma de tinta sobre los **píxeles renderizados** de la región. La señal de "imagen incrustada" solo es útil en PDFs nativos (página con texto + imagen pequeña pegada en la zona de firma).
- Región de firma por defecto (fracciones del rect de la página): `x ∈ [0.00, 0.55]`, `y ∈ [0.60, 1.00]`.
- DPI de render por defecto: 200.

## Estructura de ficheros

- Create: `src/detector_firma.py` — núcleo: config, render, recorte, scoring, orquestador `detectar()`.
- Create: `tests/test_detector_firma.py` — tests unitarios (pytest) con imágenes/PDFs sintéticos.
- Create: `tests/test_fiabilidad.py` — banco de pruebas/reporte sobre carpetas reales (script ejecutable, no pytest).
- Modify: `requirements.txt` — añadir `pytest` para los tests unitarios.

---

### Task 1: Dependencia de test (pytest)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Añadir pytest a requirements.txt**

Añadir bajo la sección de núcleo:
```
pytest>=8.0
```

- [ ] **Step 2: Instalar**

Run: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
Expected: pytest instalado correctamente.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "test: añade pytest a dependencias"
```

---

### Task 2: Config + render de la última página

**Files:**
- Create: `src/detector_firma.py`
- Test: `tests/test_detector_firma.py`

- [ ] **Step 1: Escribir el test que falla**

```python
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
```

- [ ] **Step 2: Ejecutar el test para verque falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py::test_render_ultima_pagina_devuelve_gris -v`
Expected: FAIL (módulo sin `render_ultima_pagina`).

- [ ] **Step 3: Implementación mínima**

```python
# src/detector_firma.py
"""Detección de firma manuscrita en Notas de Encargo (Fase 1)."""
import fitz
import cv2
import numpy as np

# --- Configuración ajustable ---
DPI = 200
# Región de firma como fracciones del rect de la página: (x0, y0, x1, y1)
REGION = (0.00, 0.60, 0.55, 1.00)
UMBRAL_INK = 0.010   # ratio de píxeles de tinta para considerar "firmado" (se afina con el banco de pruebas)
AREA_MIN_TRAZO = 40  # área mínima (px) de un componente para contar como trazo


def render_ultima_pagina(pdf_path, dpi=DPI):
    """Devuelve (imagen_gris_uint8, rect_pagina) de la última página."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[-1]
        rect = page.rect
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return img.copy(), rect
    finally:
        doc.close()
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py::test_render_ultima_pagina_devuelve_gris -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/detector_firma.py tests/test_detector_firma.py
git commit -m "feat: render de la última página en escala de grises"
```

---

### Task 3: Recorte de la región de firma

**Files:**
- Modify: `src/detector_firma.py`
- Test: `tests/test_detector_firma.py`

- [ ] **Step 1: Escribir el test que falla**

```python
def test_recortar_region_inferior_izquierda():
    img = np.full((1000, 800), 255, dtype=np.uint8)  # alto=1000, ancho=800
    sub, bbox = df.recortar_region(img, (0.0, 0.60, 0.55, 1.0))
    x0, y0, x1, y1 = bbox
    assert (x0, y0) == (0, 600)
    assert (x1, y1) == (440, 1000)
    assert sub.shape == (400, 440)
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py::test_recortar_region_inferior_izquierda -v`
Expected: FAIL (sin `recortar_region`).

- [ ] **Step 3: Implementación mínima**

```python
def recortar_region(img, region=REGION):
    """Recorta una región (fracciones x0,y0,x1,y1) de una imagen gris.
    Devuelve (subimagen, (px0, py0, px1, py1))."""
    h, w = img.shape[:2]
    fx0, fy0, fx1, fy1 = region
    x0, y0 = int(w * fx0), int(h * fy0)
    x1, y1 = int(w * fx1), int(h * fy1)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py::test_recortar_region_inferior_izquierda -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/detector_firma.py tests/test_detector_firma.py
git commit -m "feat: recorte de la región de firma"
```

---

### Task 4: Scoring de tinta manuscrita

**Files:**
- Modify: `src/detector_firma.py`
- Test: `tests/test_detector_firma.py`

- [ ] **Step 1: Escribir los tests que fallan**

```python
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
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py -k score_tinta -v`
Expected: FAIL (sin `score_tinta`).

- [ ] **Step 3: Implementación mínima**

```python
def score_tinta(region_img):
    """Mide presencia de tinta manuscrita en una región gris.
    Devuelve dict con ink_ratio, n_trazos."""
    # Binariza: tinta (oscuro) -> 255
    binaria = cv2.adaptiveThreshold(
        region_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 15,
    )
    total = region_img.shape[0] * region_img.shape[1]
    ink = int(np.count_nonzero(binaria))
    ink_ratio = ink / total if total else 0.0

    # Componentes conectados: cuenta "trazos" significativos
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    n_trazos = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= AREA_MIN_TRAZO))

    return {"ink_ratio": round(ink_ratio, 5), "n_trazos": n_trazos}
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py -k score_tinta -v`
Expected: PASS (ambos)

- [ ] **Step 5: Commit**

```bash
git add src/detector_firma.py tests/test_detector_firma.py
git commit -m "feat: scoring de tinta manuscrita"
```

---

### Task 5: Señal de imagen incrustada (PDFs nativos)

**Files:**
- Modify: `src/detector_firma.py`
- Test: `tests/test_detector_firma.py`

- [ ] **Step 1: Escribir el test que falla**

```python
def test_imagen_en_region_falsa_sin_imagenes(tmp_path):
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "Texto sin imagenes")
    ruta = str(tmp_path / "sintexto.pdf")
    doc.save(ruta); doc.close()

    doc2 = fitz.open(ruta)
    page2 = doc2[-1]
    assert df.imagen_en_region(page2, REGION) is False
    doc2.close()
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py::test_imagen_en_region_falsa_sin_imagenes -v`
Expected: FAIL (sin `imagen_en_region`).

- [ ] **Step 3: Implementación mínima**

```python
def imagen_en_region(page, region=REGION):
    """True si hay una imagen raster colocada dentro de la región de firma,
    y la página NO es un escaneo de página completa (tiene texto).
    En escaneos (sin texto) esta señal no es fiable y devuelve False."""
    texto = page.get_text().strip()
    if not texto:
        return False  # escaneo: toda la página es imagen; señal no aplicable
    rect = page.rect
    fx0, fy0, fx1, fy1 = region
    zona = fitz.Rect(rect.width * fx0, rect.height * fy0,
                     rect.width * fx1, rect.height * fy1)
    for img in page.get_image_info():
        bbox = fitz.Rect(img["bbox"])
        if zona.intersects(bbox):
            return True
    return False
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py::test_imagen_en_region_falsa_sin_imagenes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/detector_firma.py tests/test_detector_firma.py
git commit -m "feat: señal de imagen incrustada para PDFs nativos"
```

---

### Task 6: Orquestador `detectar()` + manejo de errores + debug

**Files:**
- Modify: `src/detector_firma.py`
- Test: `tests/test_detector_firma.py`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_detectar_firmado_con_trazo(tmp_path):
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    page = doc.new_page(width=595, height=842)
    # trazo tipo firma en zona inferior izquierda
    page.draw_bezier(fitz.Rect(0,0,0,0).tl if False else fitz.Point(40, 760),
                     fitz.Point(120, 700), fitz.Point(200, 800), fitz.Point(280, 740),
                     color=(0, 0, 0), width=2)
    page.draw_line(fitz.Point(40, 790), fitz.Point(300, 795), color=(0,0,0), width=2)
    ruta = str(tmp_path / "firmado.pdf")
    doc.save(ruta); doc.close()

    r = df.detectar(ruta)
    assert r["error"] is False
    assert r["firmado"] is True
    assert 0.0 <= r["confianza"] <= 1.0
    assert r["pagina"] == 2

def test_detectar_no_firmado_pagina_blanca(tmp_path):
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)  # última en blanco
    ruta = str(tmp_path / "blanco.pdf")
    doc.save(ruta); doc.close()
    r = df.detectar(ruta)
    assert r["error"] is False
    assert r["firmado"] is False

def test_detectar_pdf_inexistente():
    r = df.detectar("no_existe_xyz.pdf")
    assert r["error"] is True
    assert "mensaje" in r
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py -k detectar -v`
Expected: FAIL (sin `detectar`).

- [ ] **Step 3: Implementación mínima**

```python
import os

def detectar(pdf_path, debug_dir=None):
    """Detecta firma manuscrita en la última página de un PDF.
    Devuelve un dict estructurado; nunca lanza excepción por PDF inválido."""
    if not os.path.exists(pdf_path):
        return {"error": True, "mensaje": f"No existe el archivo: {pdf_path}"}
    try:
        doc = fitz.open(pdf_path)
        n_pag = doc.page_count
        page = doc[-1]
        tiene_imagen = imagen_en_region(page, REGION)
        doc.close()

        img, _rect = render_ultima_pagina(pdf_path, dpi=DPI)
        region_img, bbox = recortar_region(img, REGION)
        st = score_tinta(region_img)

        firmado_tinta = st["ink_ratio"] >= UMBRAL_INK and st["n_trazos"] >= 1
        firmado = bool(firmado_tinta or tiene_imagen)
        if tiene_imagen and not firmado_tinta:
            metodo = "imagen"
        elif firmado_tinta:
            metodo = "tinta"
        else:
            metodo = "ninguno"
        confianza = round(min(1.0, st["ink_ratio"] / UMBRAL_INK), 3) if UMBRAL_INK else 0.0
        if tiene_imagen:
            confianza = max(confianza, 0.6)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            nombre = os.path.splitext(os.path.basename(pdf_path))[0]
            cv2.imwrite(os.path.join(debug_dir, f"{nombre}_region.png"), region_img)

        return {
            "error": False,
            "firmado": firmado,
            "confianza": confianza,
            "metodo": metodo,
            "pagina": n_pag,
            "region": list(bbox),
            "detalles": {**st, "imagen_en_region": tiene_imagen},
        }
    except Exception as e:
        return {"error": True, "mensaje": f"Error procesando PDF: {e}"}
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py -k detectar -v`
Expected: PASS (los tres)

- [ ] **Step 5: Ejecutar TODA la suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector_firma.py -v`
Expected: todos PASS

- [ ] **Step 6: Commit**

```bash
git add src/detector_firma.py tests/test_detector_firma.py
git commit -m "feat: orquestador detectar() con manejo de errores y debug"
```

---

### Task 7: Banco de pruebas de fiabilidad sobre muestras reales

**Files:**
- Create: `tests/test_fiabilidad.py`

- [ ] **Step 1: Escribir el script de reporte**

```python
# tests/test_fiabilidad.py
"""Corre el detector sobre las carpetas reales y reporta fiabilidad.
Uso: .venv/Scripts/python.exe tests/test_fiabilidad.py
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fitz
import detector_firma as df

BASE = r"C:\Users\andres\Documents\Nota encargo - catastro"
FIRMADA = os.path.join(BASE, "firmada")
NO_FIRMADA = os.path.join(BASE, "no firmada")
DEBUG = os.path.join(os.path.dirname(__file__), "debug")
SINTET = os.path.join(os.path.dirname(__file__), "sinteticos_no_firmada")


def generar_negativos_sinteticos():
    """Crea versiones 'no firmadas' pintando en blanco la región de firma
    de cada PDF firmado (región geométrica fija, no el trazo detectado)."""
    os.makedirs(SINTET, exist_ok=True)
    rutas = []
    for p in sorted(glob.glob(os.path.join(FIRMADA, "*.pdf"))):
        try:
            doc = fitz.open(p)
            page = doc[-1]
            r = page.rect
            fx0, fy0, fx1, fy1 = df.REGION
            zona = fitz.Rect(r.width*fx0, r.height*fy0, r.width*fx1, r.height*fy1)
            page.add_redact_annot(zona, fill=(1, 1, 1))
            page.apply_redactions()
            destino = os.path.join(SINTET, os.path.basename(p))
            doc.save(destino)
            doc.close()
            rutas.append(destino)
        except Exception as e:
            print("  (no se pudo sintetizar", os.path.basename(p), ":", e, ")")
    return rutas


def correr(rutas, etiqueta_esperada):
    print(f"\n=== {etiqueta_esperada.upper()} ({len(rutas)} docs) ===")
    aciertos = 0
    for p in rutas:
        r = df.detectar(p, debug_dir=DEBUG)
        if r["error"]:
            print(f"  ERROR {os.path.basename(p)[:45]:45} -> {r['mensaje']}")
            continue
        ok = (r["firmado"] == (etiqueta_esperada == "firmada"))
        aciertos += int(ok)
        marca = "OK " if ok else "XX "
        d = r["detalles"]
        print(f"  {marca}{os.path.basename(p)[:42]:42} firmado={str(r['firmado']):5} "
              f"conf={r['confianza']:.2f} ink={d['ink_ratio']:.4f} trazos={d['n_trazos']} "
              f"metodo={r['metodo']}")
    print(f"  -> aciertos: {aciertos}/{len(rutas)}")
    return aciertos, len(rutas)


def main():
    firmadas = sorted(glob.glob(os.path.join(FIRMADA, "*.pdf")))
    no_firmadas = sorted(glob.glob(os.path.join(NO_FIRMADA, "*.pdf")))
    print(f"REGION={df.REGION}  UMBRAL_INK={df.UMBRAL_INK}  DPI={df.DPI}")
    correr(firmadas, "firmada")
    if no_firmadas:
        correr(no_firmadas, "no firmada")
    else:
        print("\n(no hay 'no firmada' reales; usando negativos sintéticos)")
        sint = generar_negativos_sinteticos()
        correr(sint, "no firmada")
    print(f"\nRecortes de debug en: {DEBUG}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Ejecutar el banco de pruebas**

Run: `.venv/Scripts/python.exe tests/test_fiabilidad.py`
Expected: imprime tabla por documento + aciertos sobre firmadas y sobre negativos sintéticos. (Los números guiarán el ajuste de `UMBRAL_INK` y `REGION`.)

- [ ] **Step 3: Revisar visualmente los recortes**

Abrir `tests/debug/*_region.png` y confirmar que la región recortada cae sobre la zona de la firma. Si no, ajustar `REGION` en `src/detector_firma.py` y reejecutar.

- [ ] **Step 4: Commit**

```bash
git add tests/test_fiabilidad.py
git commit -m "test: banco de fiabilidad sobre muestras reales + negativos sintéticos"
```

---

## Ajuste y criterio de cierre (Fase 1)

Tras correr el banco de pruebas:
1. Observar la distribución de `ink_ratio` entre firmadas y negativos sintéticos.
2. Fijar `UMBRAL_INK` en un valor que separe ambas clases (idealmente 9/9 firmadas detectadas y negativos sintéticos correctamente "no firmados").
3. Validar visualmente los recortes de debug (que la región sea la correcta).
4. El usuario revisa los resultados. Si los considera fiables sobre su muestra → luz verde para Fase 2 (microservicio FastAPI) y Fase 3 (integración n8n con arreglo del recorte de última página).

## Self-review

- **Cobertura del spec:** render (T2), recorte región (T3), señal tinta (T4), señal imagen (T5), orquestador + errores + debug (T6), banco de pruebas + negativos sintéticos (T7). ✔
- **Sin placeholders:** todos los pasos llevan código/comando concretos. ✔
- **Consistencia de tipos:** `render_ultima_pagina` → (img, rect); `recortar_region` → (sub, bbox); `score_tinta` → {ink_ratio, n_trazos}; `imagen_en_region` → bool; `detectar` consume todos. Nombres consistentes entre tareas. ✔
- Fuera de alcance respetado: firma digital y n8n quedan para fases posteriores.
