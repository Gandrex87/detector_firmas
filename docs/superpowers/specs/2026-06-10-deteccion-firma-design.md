# Detección de firma manuscrita en Notas de Encargo — Diseño

Fecha: 2026-06-10
Estado: aprobado (diseño), pendiente de implementación

## Problema

El flujo n8n `Validador NOTA_DE_ENCARGO - CATASTRO` clasifica un PDF como
`NOTA_DE_ENCARGO` o `CATASTRO`, pero no verifica si el documento viene **firmado**.
En el sticky note del flujo queda anotado el pendiente: `VALIDAR FIRMA`.

Objetivo de esta fase: detectar de forma fiable si una **Nota de Encargo** trae
**firma manuscrita**, y medir esa fiabilidad sobre documentos reales. Las firmas
**digitales/criptográficas** quedan para una fase posterior (solo si esta funciona).

## Supuestos confirmados con el usuario

- La firma está en la **última página**, en la **zona inferior izquierda**.
- Solo se dispone de documentos **firmados** ahora mismo
  (`C:\Users\andres\Documents\Nota encargo - catastro\firmada\`, 9 PDFs).
  La carpeta `no firmada\` está vacía.
- Python 3.13 disponible localmente; sin librerías de PDF/visión instaladas todavía.

## Enfoque elegido

**Opción 1 — Heurística de visión por computadora (OpenCV + PyMuPDF).**
Se descarta por ahora el VLM (Opción 2) y el modelo dedicado YOLO (Opción 3); el
modelo dedicado queda como mejora futura si la heurística no alcanza la fiabilidad
deseada.

Razón: la "pista de región" (última página, inferior izquierda) es fuerte y permite
una heurística explicable, sin GPU, que corre en local desde el primer día y produce
un score numérico ajustable.

## Arquitectura

Tres componentes con responsabilidad única:

### 1. Núcleo de detección — `src/detector_firma.py`

Entrada: ruta a un PDF (o bytes). Salida: dict estructurado.

Pasos:
1. Abrir el PDF con PyMuPDF y seleccionar la **última página**.
2. Renderizar esa página a imagen a una DPI razonable (~200).
3. Recortar la **región de firma**, configurable; por defecto:
   `x ∈ [0%, 55%]`, `y ∈ [60%, 100%]` (inferior izquierda).
4. Combinar dos señales:
   - **(a) Tinta manuscrita:** binarización adaptativa → componentes conectados →
     filtrar por geometría de trazo (descartar lo que parece texto impreso:
     muy pequeño, alineado en filas regulares) → calcular `ink_score`.
   - **(b) Imagen incrustada:** inspeccionar con PyMuPDF si hay una imagen raster
     colocada dentro de la región de firma (firmas pegadas como imagen).
5. Decidir `firmado` por umbral sobre la señal combinada.

Salida:
```json
{
  "firmado": true,
  "confianza": 0.0,
  "metodo": "tinta | imagen | ninguno",
  "pagina": 3,
  "region": [x0, y0, x1, y1],
  "detalles": { "ink_score": 0.0, "imagen_en_region": false }
}
```
Opción de depuración: guardar un PNG del recorte analizado para inspección visual.

Parámetros configurables (región, DPI, umbral) viven en un único bloque de
constantes / objeto de configuración, para poder afinarlos sin tocar la lógica.

### 2. Banco de pruebas — `tests/test_fiabilidad.py`

- Recorre `firmada/` y `no firmada/`, ejecuta el detector sobre cada PDF.
- Imprime score por archivo + métricas resumen (recall sobre firmadas; y cuando
  haya negativos, tasa de falsos positivos).
- **Negativos sintéticos:** genera versiones "no firmadas" pintando en blanco la
  región de firma de cada PDF firmado, para comprobar que el detector responde
  correctamente "no firmado" cuando la zona está vacía. (Se borra una región
  geométrica fija, no el trazo detectado, para no ser circular.)
- Guarda recortes de debug en `tests/debug/` para revisión visual.

### 3. (Fase 2) Microservicio — `src/servicio.py`

Solo si la fiabilidad es buena. FastAPI, mismo patrón que los servicios internos
existentes (`/recortar-pdf` en `10.1.0.188:8020`, `/ocr` en `10.1.0.182:7862`):
`POST /verificar-firma` recibe el PDF (multipart) y devuelve el JSON del núcleo.

## Flujo de datos

```
PDF → detector_firma.detectar(pdf)
        → última página → render → recorte región firma
        → señal tinta + señal imagen → confianza → {firmado, ...}
```

## Manejo de errores

- PDF ilegible / corrupto → `{ "error": true, "mensaje": ... }`, no excepción cruda.
- PDF sin páginas → error controlado.
- Región fuera de límites en páginas muy pequeñas → recortar a límites válidos.

## Pruebas y criterio de éxito

- **Fase 1:** sobre las 9 firmadas, recall objetivo alto (idealmente 9/9 detectadas).
  Sobre los negativos sintéticos, que el detector diga "no firmado".
- Revisión visual de los recortes de debug para validar que mira la zona correcta.
- Afinar umbral según la distribución de scores observada.
- Criterio para pasar a Fase 2: el usuario revisa los resultados y los considera
  fiables sobre su muestra real.

## Dependencias

- `pymupdf` (render + inspección de imágenes incrustadas; no requiere poppler)
- `opencv-python`, `numpy` (análisis de imagen)
- (Fase 2) `fastapi`, `uvicorn`, `python-multipart`

## Fuera de alcance (esta fase)

- Firmas digitales/criptográficas (PAdES, certificados) → fase posterior.
- Identidad del firmante.
- Integración en n8n (incluye arreglar el recorte para abarcar la última página)
  → Fase 3, tras validar fiabilidad.
