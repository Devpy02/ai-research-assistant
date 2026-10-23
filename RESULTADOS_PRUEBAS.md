# Evidencias de pruebas — Entrega 2, Parte 1

## Pruebas automatizadas

Comando ejecutado desde la raíz del proyecto:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
```

Resultado en el entorno de construcción: **9 pruebas aprobadas, 0 fallidas**.

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

Nota: algunos comportamientos se validan conjuntamente dentro de una misma prueba; por eso hay 9 métodos de prueba para los 10 puntos enumerados.

## Prueba de integración reproducible

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

## Límite de la evidencia

Las pruebas se ejecutaron en el entorno usado para construir este paquete. Antes de aceptar la entrega en el equipo de desarrollo Windows del proyecto, se debe ejecutar el mismo conjunto de pruebas allí. No se midió aún rendimiento de embeddings ni generación porque este hito se limita intencionadamente a la representación, fragmentación y persistencia documental.
