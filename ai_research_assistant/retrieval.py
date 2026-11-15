
"""Recuperación semántica sobre el índice vectorial local."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ai_research_assistant.embeddings import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    embed_texts,
    get_model_info,
)
from ai_research_assistant.storage import load_index
from ai_research_assistant.vector_storage import load_vector_index


class RetrievalError(ValueError):
    """Error durante la recuperación semántica."""


def search(
    query: str,
    *,
    index_dir: Path,
    vector_dir: Path,
    top_k: int = 3,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Busca los fragmentos más similares semánticamente a una consulta."""

    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("La consulta debe ser un texto no vacío")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalError("top_k debe ser un entero positivo")

    # Cargar y validar el índice documental.
    manifest, chunks = load_index(Path(index_dir))

    # Calcular la identidad exacta del índice documental.
    manifest_path = Path(index_dir).resolve() / "manifest.json"
    source_manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    source_chunks_sha256 = manifest["chunks_sha256"]

    # Obtener la identidad del modelo de embeddings.
    model_info = get_model_info(
        base_url=base_url,
        model=model,
        timeout=timeout,
    )

    # Cargar los vectores y validar su correspondencia documental.
    _, matrix = load_vector_index(
        Path(vector_dir),
        chunks,
        model_info,
        source_manifest_sha256=source_manifest_sha256,
        source_chunks_sha256=source_chunks_sha256,
    )

    # Convertir la consulta en un vector.
    query_vectors = embed_texts(
        [query.strip()],
        base_url=base_url,
        model=model,
        timeout=timeout,
        expected_dimension=matrix.shape[1],
    )

    query_vector = np.asarray(query_vectors[0], dtype=np.float32)

    # Calcular similitud coseno.
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    if query_norm == 0 or np.any(matrix_norms == 0):
        raise RetrievalError(
            "No se puede calcular la similitud coseno con vectores nulos"
        )

    similarities = (matrix @ query_vector) / (matrix_norms * query_norm)

    # Ordenar de mayor a menor similitud.
    ranked_indices = np.argsort(-similarities, kind="stable")[:top_k]

    results = []

    for position in ranked_indices:
        chunk = chunks[int(position)]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "source_path": chunk["source_path"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "score": float(similarities[position]),
            }
        )

    return results
