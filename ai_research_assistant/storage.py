"""Persistencia JSONL con validación de integridad y trazabilidad."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .chunking import build_records, sha256_text

SCHEMA_VERSION = 1
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def persist_documents(
    source: Path,
    output_dir: Path,
    *,
    chunk_size: int = 400,
    overlap: int = 60,
) -> dict:
    """Fragmenta y persiste metadatos/documentos en JSONL + manifiesto."""
    documents, chunks = build_records(source, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No se generaron fragmentos; no se escribió ningún índice")

    output_dir = output_dir.resolve()
    chunks_path = output_dir / CHUNKS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    chunks_bytes = "".join(_canonical_json(row) + "\n" for row in chunks).encode("utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "storage_format": "jsonl",
        "chunking": {
            "method": "whitespace_token_window",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "offset_unit": "unicode_character",
            "offset_convention": "start_inclusive_end_exclusive",
        },
        "documents": documents,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_file": CHUNKS_FILENAME,
        "chunks_sha256": _sha256_bytes(chunks_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    # Cada archivo se sustituye atómicamente. load_index valida el hash y detecta
    # cualquier interrupción que dejase manifiesto y JSONL de versiones distintas.
    _atomic_write(chunks_path, chunks_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return manifest


def load_index(output_dir: Path) -> tuple[dict, list[dict]]:
    """Carga el índice y valida hash, esquema, IDs y correspondencia documental."""
    output_dir = output_dir.resolve()
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe el manifiesto: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Versión de esquema no soportada: {manifest.get('schema_version')}")

    chunks_path = output_dir / manifest.get("chunks_file", CHUNKS_FILENAME)
    raw = chunks_path.read_bytes()
    if _sha256_bytes(raw) != manifest.get("chunks_sha256"):
        raise ValueError("El hash de chunks.jsonl no coincide con manifest.json; índice alterado o inconsistente")

    chunks: list[dict] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido en {chunks_path}, línea {line_number}") from exc

    if len(chunks) != manifest.get("chunk_count"):
        raise ValueError("La cantidad de fragmentos no coincide con el manifiesto")
    ids = [row.get("chunk_id") for row in chunks]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Hay IDs de fragmento ausentes o duplicados")

    documents = {row["document_id"]: row for row in manifest.get("documents", [])}
    if len(documents) != manifest.get("document_count"):
        raise ValueError("La cantidad de documentos no coincide con el manifiesto")
    counts: dict[str, int] = {document_id: 0 for document_id in documents}
    for row in chunks:
        document_id = row.get("document_id")
        if document_id not in documents:
            raise ValueError(f"Fragmento asociado a un documento desconocido: {document_id}")
        doc = documents[document_id]
        if row.get("source_path") != doc.get("source_path"):
            raise ValueError(f"Procedencia inconsistente en fragmento {row.get('chunk_id')}")
        if row.get("content_sha256") != doc.get("content_sha256"):
            raise ValueError(f"Hash de documento inconsistente en fragmento {row.get('chunk_id')}")
        start, end, text = row.get("char_start"), row.get("char_end"), row.get("text")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(text, str) or end <= start:
            raise ValueError(f"Offsets inválidos en fragmento {row.get('chunk_id')}")
        if len(text) != end - start:
            raise ValueError(f"Longitud y offsets no coinciden en fragmento {row.get('chunk_id')}")
        counts[document_id] += 1
    for document_id, doc in documents.items():
        if counts[document_id] != doc.get("chunk_count"):
            raise ValueError(f"Cantidad de fragmentos incorrecta para {doc.get('source_path')}")
    return manifest, chunks


def verify_sources(output_dir: Path, source: Path) -> list[str]:
    """Compara contenido actual de origen con los hashes guardados en el manifiesto."""
    manifest, _ = load_index(output_dir)
    source = source.resolve()
    if source.is_file():
        root = source.parent
        source_map = {source.name: source}
    elif source.is_dir():
        root = source
        source_map = {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".markdown"}
        }
    else:
        raise FileNotFoundError(f"No existe el origen: {source}")

    errors: list[str] = []
    for doc in manifest.get("documents", []):
        relative_path = doc["source_path"]
        path = source_map.get(relative_path)
        if path is None:
            errors.append(f"Falta documento de origen: {relative_path}")
            continue
        try:
            current = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"No se pudo leer {relative_path}: {exc}")
            continue
        if sha256_text(current) != doc["content_sha256"]:
            errors.append(f"El contenido cambió desde la indexación: {relative_path}")
    indexed_paths = {doc["source_path"] for doc in manifest.get("documents", [])}
    for relative_path in sorted(set(source_map) - indexed_paths):
        errors.append(f"Documento nuevo no incluido en el índice: {relative_path}")
    return errors
