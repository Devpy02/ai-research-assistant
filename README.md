# AI Research Assistant — Entrega 2, Parte 1

Núcleo documental inicial: lectura de documentos TXT/Markdown, fragmentación determinista, trazabilidad y persistencia JSONL con manifiesto de integridad. Esta parte **no integra todavía embeddings, recuperación semántica ni generación con Qwen**, de acuerdo con el punto de control solicitado por el cliente.

## Requisitos

- Python 3.11 o compatible.
- Solo biblioteca estándar de Python; no instala dependencias.

## Estructura

```text
ai_research_assistant_entrega2/
├── ai_research_assistant/
│   ├── __init__.py
│   ├── chunking.py
│   ├── storage.py
│   └── cli.py
├── examples/documentacion_demo.md
├── tests/
│   ├── test_chunking.py
│   └── test_storage.py
└── data/
    ├── documents/
    └── index/
```

## Diseño técnico

- **Formatos admitidos:** `.txt`, `.md` y `.markdown`, codificados en UTF-8.
- **Fragmentación:** ventanas de 400 unidades `\S+` (palabras aproximadas) con solapamiento de 60. Se puede configurar por CLI. No se afirma que estas unidades equivalgan a los tokens del modelo.
- **Offsets:** índices de caracteres Unicode, inicio inclusivo y fin exclusivo. El texto guardado coincide exactamente con el segmento indicado por los offsets.
- **Trazabilidad:** cada fragmento contiene `chunk_id`, `document_id`, `source_path`, `filename`, `chunk_index`, offsets, rango de palabras aproximadas, hash del documento y texto.
- **IDs:** deterministas para el mismo documento (ruta relativa + hash del contenido) y el mismo corte. Si cambia el contenido, cambia el ID del documento y de sus fragmentos.
- **Persistencia:** `chunks.jsonl` guarda un objeto JSON por línea; `manifest.json` registra parámetros, documentos, cantidades y SHA-256 del JSONL.
- **Integridad:** al cargar se valida el hash del JSONL, la versión de esquema, unicidad de IDs, referencias a documentos, hashes y offsets. `verify` también compara los hashes del origen actual.
- **Escritura:** archivos temporales y sustitución atómica individual; si una interrupción deja archivos de versiones distintas, la comprobación de hash detecta la inconsistencia.

## Uso en Windows PowerShell

Desde la carpeta raíz del proyecto:

```powershell
python -m ai_research_assistant.cli ingest --source .\examples\documentacion_demo.md --output .\data\index --chunk-size 25 --overlap 5
python -m ai_research_assistant.cli show --index .\data\index --limit 10
python -m ai_research_assistant.cli verify --source .\examples\documentacion_demo.md --index .\data\index
```

Para indexar un directorio con varios documentos:

```powershell
python -m ai_research_assistant.cli ingest --source .\data\documents --output .\data\index
python -m ai_research_assistant.cli verify --source .\data\documents --index .\data\index
```

El directorio de documentos debe contener al menos un archivo TXT/MD no vacío. No se incluyen aún PDF ni DOCX, para evitar dependencias y mantener acotado este primer hito.

## Pruebas automatizadas

Desde la raíz del proyecto:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
```

Las pruebas cubren solapamiento, offsets exactos, configuración inválida, estabilidad de IDs, persistencia y recarga, trazabilidad, detección de cambios en el origen, integridad del JSONL y rechazo de documentos vacíos.

## Alcance pendiente

Esta entrega parcial no genera embeddings ni contiene `embeddings.npy`, porque el punto de revisión actual exige aprobar primero la representación y persistencia de fragmentos. En el siguiente hito se podrá añadir una matriz NumPy alineada con el orden de los fragmentos, acompañada de metadatos del modelo de embeddings y verificaciones de correspondencia.
