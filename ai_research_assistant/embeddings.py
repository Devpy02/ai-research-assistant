
"""Generación y validación de embeddings mediante la API local de Ollama."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text:latest"
DEFAULT_TIMEOUT = 120
DEFAULT_BATCH_SIZE = 4


class EmbeddingError(RuntimeError):
    """Error durante la generación o validación de embeddings."""


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    """Envía una petición JSON y devuelve la respuesta."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EmbeddingError(
            f"Ollama respondió con HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise EmbeddingError(
            f"No se pudo conectar con Ollama en {url}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise EmbeddingError(
            f"Tiempo de espera agotado al consultar Ollama: {url}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingError(
            "Ollama devolvió una respuesta que no es JSON válido"
        ) from exc

    if not isinstance(result, dict):
        raise EmbeddingError("La respuesta de Ollama no es un objeto JSON")

    return result


def _get_json(
    url: str,
    *,
    timeout: int,
) -> dict[str, Any]:
    """Realiza una petición GET y devuelve la respuesta JSON."""
    request = Request(url, method="GET")

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EmbeddingError(
            f"Ollama respondió con HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise EmbeddingError(
            f"No se pudo conectar con Ollama en {url}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise EmbeddingError(
            f"Tiempo de espera agotado al consultar Ollama: {url}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingError(
            "Ollama devolvió una respuesta que no es JSON válido"
        ) from exc

    if not isinstance(result, dict):
        raise EmbeddingError("La respuesta de Ollama no es un objeto JSON")

    return result


def get_model_info(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Obtiene la identidad y los metadatos del modelo instalado."""
    response = _get_json(
        f"{base_url.rstrip('/')}/api/tags",
        timeout=timeout,
    )

    models = response.get("models")

    if not isinstance(models, list):
        raise EmbeddingError(
            "La respuesta de Ollama no contiene una lista de modelos"
        )

    model_data = next(
        (
            item for item in models
            if isinstance(item, dict)
            and item.get("name") == model
        ),
        None,
    )

    if model_data is None:
        raise EmbeddingError(
            f"El modelo '{model}' no está instalado en Ollama"
        )

    details = model_data.get("details") or {}

    if not isinstance(details, dict):
        raise EmbeddingError("Los metadatos del modelo son inválidos")

    digest = model_data.get("digest")
    dimension = details.get("embedding_length")

    if not isinstance(digest, str) or not digest:
        raise EmbeddingError("Ollama no proporcionó el digest del modelo")

    if not isinstance(dimension, int) or dimension <= 0:
        raise EmbeddingError(
            "Ollama no proporcionó una dimensión de embedding válida"
        )

    return {
        "name": model_data["name"],
        "digest": digest,
        "family": details.get("family"),
        "format": details.get("format"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "embedding_length": dimension,
        "context_length": details.get("context_length"),
        "modified_at": model_data.get("modified_at"),
        "size": model_data.get("size"),
        "capabilities": model_data.get("capabilities", []),
    }


def _validate_embeddings(
    embeddings: Any,
    *,
    expected_count: int,
    expected_dimension: int | None = None,
) -> int:
    """Valida cantidad, dimensiones y valores numéricos de los vectores."""
    if not isinstance(embeddings, list):
        raise EmbeddingError("La respuesta no contiene una lista de embeddings")

    if len(embeddings) != expected_count:
        raise EmbeddingError(
            f"Se esperaban {expected_count} embeddings y se recibieron "
            f"{len(embeddings)}"
        )

    dimension = None

    for index, vector in enumerate(embeddings):
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError(f"Embedding vacío o inválido en posición {index}")

        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise EmbeddingError(
                f"Dimensión inconsistente en embedding {index}: "
                f"{len(vector)} frente a {dimension}"
            )

        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingError(
                    f"Valor no numérico en embedding {index}"
                )
            if not math.isfinite(value):
                raise EmbeddingError(
                    f"Valor no finito en embedding {index}"
                )

    if expected_dimension is not None and dimension != expected_dimension:
        raise EmbeddingError(
            f"Dimensión inesperada: {dimension}; "
            f"se esperaba {expected_dimension}"
        )

    return dimension


def embed_texts(
    texts: list[str],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    expected_dimension: int | None = None,
) -> list[list[float]]:
    """Genera embeddings para una lista de textos."""
    if not texts:
        raise EmbeddingError("La lista de textos está vacía")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise EmbeddingError("Todos los textos deben ser cadenas no vacías")

    response = _post_json(
        f"{base_url.rstrip('/')}/api/embed",
        {"model": model, "input": texts},
        timeout=timeout,
    )

    if response.get("model") != model:
        raise EmbeddingError(
            f"Modelo inesperado en la respuesta: {response.get('model')}"
        )

    embeddings = response.get("embeddings")
    _validate_embeddings(
        embeddings,
        expected_count=len(texts),
        expected_dimension=expected_dimension,
    )

    return embeddings


def embed_chunks(
    chunks: list[dict[str, Any]],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    expected_dimension: int | None = None,
) -> list[dict[str, Any]]:
    """Genera embeddings por lotes y conserva la asociación con cada chunk_id."""
    if batch_size < 1:
        raise EmbeddingError("batch_size debe ser mayor que cero")

    if not chunks:
        raise EmbeddingError("No hay fragmentos para procesar")

    seen_ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EmbeddingError(f"chunk_id ausente o inválido en posición {index}")
        if chunk_id in seen_ids:
            raise EmbeddingError(f"chunk_id duplicado: {chunk_id}")
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError(f"Texto vacío o inválido en fragmento {chunk_id}")

        seen_ids.add(chunk_id)

    results: list[dict[str, Any]] = []
    dimension = expected_dimension

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch]

        vectors = embed_texts(
            texts,
            base_url=base_url,
            model=model,
            timeout=timeout,
            expected_dimension=dimension,
        )

        batch_dimension = len(vectors[0])
        if dimension is None:
            dimension = batch_dimension

        for chunk, vector in zip(batch, vectors, strict=True):
            results.append({
                "chunk_id": chunk["chunk_id"],
                "embedding": vector,
            })

    if len(results) != len(chunks):
        raise EmbeddingError("No se generó un vector para cada fragmento")

    return results
