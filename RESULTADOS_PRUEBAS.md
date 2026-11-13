# Evidencias de pruebas — Entrega 2, Parte 1

## 1. Pruebas automatizadas en el entorno de construcción

Comando ejecutado desde la raíz del proyecto:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
```

Resultado: **9 pruebas aprobadas, 0 fallidas**.

Casos cubiertos:

1. Fragmentación con solapamiento esperado.
2. Correspondencia exacta entre texto y offsets de caracteres.
3. Texto vacío produce cero fragmentos.
4. Configuración de fragmentación inválida se rechaza.
5. IDs deterministas y sensibles a cambios de contenido.
6. Persistencia y recarga conservan procedencia y contenido.
7. Reindexar el mismo documento produce los mismos IDs.
8. Cambios en el documento de origen son detectados.
9. Modificaciones en `chunks.jsonl` son detectadas mediante SHA-256.
10. Documentos vacíos son rechazados.

**Nota:** algunos comportamientos se validan conjuntamente dentro de una misma prueba; por eso hay 9 métodos de prueba para los 10 puntos enumerados.

## 2. Prueba de integración reproducible en el entorno de construcción

Se ejecutaron estos comandos contra el documento de ejemplo incluido:

```powershell
python -m ai_research_assistant.cli ingest --source .\examples\documentacion_demo.md --output .\data\index --chunk-size 25 --overlap 5
python -m ai_research_assistant.cli show --index .\data\index --limit 10
python -m ai_research_assistant.cli verify --source .\examples\documentacion_demo.md --index .\data\index
```

Resultado observado:

- 1 documento indexado.
- 4 fragmentos persistidos en `data/index/chunks.jsonl`.
- IDs únicos y offsets visibles en la salida de `show`.
- `verify` confirmó integridad del índice y coincidencia del hash del documento de origen.
- `show` cargó el índice desde disco en un proceso de CLI separado, comprobando la recarga tras finalizar el proceso de indexación.

## 3. Validación en el equipo objetivo

Se realizó una segunda validación utilizando el intérprete Python instalado en el equipo Windows de destino.

### 3.1. Entorno de ejecución

Comando:

```bash
py -3.11 --version
```

Resultado:

```text
Python 3.11.6
```

Se confirmó la disponibilidad del intérprete Python 3.11.6.

### 3.2. Ejecución de pruebas unitarias

Comando:

```bash
py -3.11 -m unittest discover -s tests -v
```

Resultado:

```text
Ran 9 tests in 0.088s

OK
```

**Resultado: 9 pruebas aprobadas, 0 fallidas.**

Los nueve métodos de prueba se ejecutaron correctamente en el equipo objetivo.

### 3.3. Prueba funcional de indexación

Para evitar modificar el índice versionado del proyecto, se utilizó un directorio temporal independiente.

Comando:

```bash
py -3.11 -m ai_research_assistant.cli ingest --source ./examples/documentacion_demo.md --output ./data/prueba_index_cli --chunk-size 25 --overlap 5
```

Resultado observado:

```json
{
  "status": "ok",
  "document_count": 1,
  "chunk_count": 4
}
```

La operación finalizó correctamente y generó los archivos `chunks.jsonl` y `manifest.json`.

### 3.4. Visualización de fragmentos

Comando:

```bash
py -3.11 -m ai_research_assistant.cli show --index ./data/prueba_index_cli --limit 10
```

Resultado observado:

- 1 documento y 4 fragmentos recuperados.
- Identificadores de fragmento visibles.
- Archivo de origen identificado correctamente.
- Índices de fragmentación y offsets de caracteres mostrados.
- Contenido textual de los cuatro fragmentos recuperado correctamente.
- Solapamiento entre fragmentos coherente con la configuración utilizada.

### 3.5. Verificación de integridad

Comando:

```bash
py -3.11 -m ai_research_assistant.cli verify --source ./examples/documentacion_demo.md --index ./data/prueba_index_cli
```

Resultado:

```text
Índice íntegro: sí | documentos: 1 | fragmentos: 4
Verificación de origen: OK; hashes de contenido coinciden.
```

La verificación confirmó la integridad del índice y la correspondencia entre el contenido persistido y el documento original.

### 3.6. Limpieza y estado del repositorio

Una vez finalizadas las pruebas, se eliminó el directorio temporal:

```bash
rm -rf ./data/prueba_index_cli
git status --short
```

El comando `git status --short` no produjo salida, lo que confirma que no quedaron modificaciones pendientes ni archivos temporales sin seguimiento en el repositorio.

## 4. Resumen de resultados

| Verificación | Resultado |
|---|---|
| Compatibilidad con Python 3.11.6 | Correcta |
| Pruebas unitarias | 9/9 aprobadas |
| Indexación mediante CLI | Correcta |
| Generación de 4 fragmentos | Correcta |
| Visualización y recuperación | Correcta |
| Integridad del índice | Correcta |
| Verificación de hashes del origen | Correcta |
| Limpieza de archivos temporales | Correcta |
| Estado final de Git | Limpio |

## 5. Conclusiones y límites de la evidencia

La implementación del núcleo documental de la Entrega 2, Parte 1 superó las pruebas unitarias y funcionales realizadas tanto en el entorno de construcción como en el equipo objetivo con Python 3.11.6.

Los resultados respaldan el funcionamiento de los componentes de fragmentación, persistencia, recuperación de fragmentos y verificación de integridad documental.

Esta validación se limita al procesamiento de archivos TXT y Markdown, la fragmentación determinista y la persistencia documental.

No se han evaluado todavía el rendimiento de embeddings, la recuperación semántica ni la generación de respuestas mediante modelos de lenguaje, ya que corresponden a etapas posteriores del proyecto.
