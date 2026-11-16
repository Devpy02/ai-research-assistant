"""Orquestación del asistente: recuperación, validación y generación local."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .embeddings import DEFAULT_BASE_URL, DEFAULT_MODEL
from .evidence_validator import validate_evidence
from .retrieval import search


GENERATION_MODEL = "qwen2.5:1.5b"


class AssistantError(RuntimeError):
    """Error controlado durante la ejecución del asistente."""


def generate_answer(
    query: str,
    *,
    index_dir: str = "data/index",
    vector_dir: str = "data/vector_index",
    base_url: str = DEFAULT_BASE_URL,
    embedding_model: str = DEFAULT_MODEL,
    generation_model: str = GENERATION_MODEL,
    top_k: int = 4,
    timeout: int = 120,
) -> dict:
    """Responde una consulta únicamente cuando la evidencia es suficiente."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    # 1. Recuperación semántica.
    retrieved = search(
        query.strip(),
        index_dir=index_dir,
        vector_dir=vector_dir,
        top_k=top_k,
        base_url=base_url,
        model=embedding_model,
        timeout=timeout,
    )

    if not retrieved:
        return {
            "status": "insufficient",
            "answer": None,
            "reason": "No se recuperaron fragmentos documentales.",
            "sources": [],
        }

    # 2. Validación de suficiencia de la evidencia.
    validation = validate_evidence(
        query.strip(),
        retrieved,
        base_url=base_url,
        model=generation_model,
        timeout=timeout,
    )

    status = validation.get("status")

    if status != "sufficient":
        return {
            "status": status or "insufficient",
            "answer": None,
            "reason": validation.get(
                "reason", "La evidencia no permite responder con seguridad."
            ),
            "sources": [],
            "retrieved_count": len(retrieved),
        }

    # 3. Solo se utilizan fragmentos citados por el validador.
    retrieved_by_id = {
        item["chunk_id"]: item
        for item in retrieved
    }

    validated_evidence = validation.get("evidence", [])
    selected_chunks = []

    for item in validated_evidence:
        chunk_id = item.get("chunk_id")
        if chunk_id in retrieved_by_id and chunk_id not in {
            chunk["chunk_id"] for chunk in selected_chunks
        }:
            selected_chunks.append(retrieved_by_id[chunk_id])

    if not selected_chunks:
        return {
            "status": "insufficient",
            "answer": None,
            "reason": "La validación no proporcionó fuentes utilizables.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in selected_chunks
    )

    prompt = f"""Responde la pregunta de forma directa y precisa utilizando exclusivamente la evidencia documental.

Pregunta:
{query.strip()}

Evidencia documental:
{context}

Instrucciones obligatorias:
1. Identifica el dato concreto que responde a la pregunta.
2. Incluye ese dato explícitamente en la respuesta. No lo sustituyas por identificadores, offsets ni metadatos.
3. Si preguntan por un endpoint, ruta, nombre, valor o procedimiento, escribe literalmente el dato que aparece en la evidencia.
4. No describas dónde está la respuesta en lugar de proporcionar la respuesta.
5. No inventes ni completes información que no aparezca en los fragmentos.
6. Redacta en español, con una o dos oraciones como máximo.
7. Añade al final la referencia del fragmento utilizado, con este formato: [Fuente: identificador].

Ejemplo:
Pregunta: ¿Qué endpoint permite generar texto?
Respuesta correcta: El endpoint /api/generate permite generar texto. [Fuente: chk_123]

Tu respuesta:
"""

    payload = {
        "model": generation_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente técnico documental. "
                    "Respondes únicamente a partir de la evidencia proporcionada."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise AssistantError(
            f"No se pudo generar la respuesta mediante Ollama: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssistantError(
            "Ollama devolvió una respuesta que no se pudo interpretar."
        ) from exc

    answer = result.get("message", {}).get("content", "").strip()

    if not answer:
        raise AssistantError("El modelo no generó una respuesta.")

    return {
        "status": "answered",
        "answer": answer,
        "reason": validation.get("reason", ""),
        "sources": [
            {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "score": chunk["score"],
            }
            for chunk in selected_chunks
        ],
        "retrieved_count": len(retrieved),
    }
