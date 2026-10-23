r"""Lectura y fragmentación determinista de documentos de texto/Markdown.

La unidad de ventana es una palabra separada por espacios (\S+), no el tokenizador
interno de Qwen o Nomic. Esta aproximación evita dependencias adicionales en esta
primera fase. Los offsets guardados son posiciones de caracteres [inicio, fin).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}
_WORD_RE = re.compile(r"\S+")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_documents(source: Path) -> tuple[Path, list[Path]]:
    """Devuelve raíz y archivos admitidos ordenados de forma reproducible."""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"No existe la ruta de origen: {source}")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Formato no admitido: {source.suffix}. Use TXT, MD o MARKDOWN."
            )
        return source.parent, [source]
    if not source.is_dir():
        raise ValueError(f"El origen no es un archivo ni un directorio: {source}")
    files = sorted(
        (p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.relative_to(source).as_posix().casefold(),
    )
    if not files:
        raise ValueError(f"No se encontraron documentos TXT/MD en: {source}")
    return source, files


def make_document_id(relative_path: str, content_sha256: str) -> str:
    """ID estable para una ruta relativa y una versión concreta del contenido."""
    raw = f"{relative_path}\0{content_sha256}".encode("utf-8")
    return "doc_" + hashlib.sha256(raw).hexdigest()[:20]


def make_chunk_id(document_id: str, chunk_index: int, char_start: int, char_end: int) -> str:
    raw = f"{document_id}:{chunk_index}:{char_start}:{char_end}".encode("utf-8")
    return "chk_" + hashlib.sha256(raw).hexdigest()[:24]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[dict[str, int | str]]:
    r"""Divide texto en ventanas deterministas con offsets de caracteres.

    `chunk_size` y `overlap` se expresan en unidades \S+ (palabras aproximadas).
    El campo `text` es exactamente text[char_start:char_end].
    """
    if chunk_size < 1:
        raise ValueError("chunk_size debe ser mayor que cero")
    if overlap < 0:
        raise ValueError("overlap no puede ser negativo")
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return []

    chunks: list[dict[str, int | str]] = []
    step = chunk_size - overlap
    start_token = 0
    chunk_index = 0
    while start_token < len(matches):
        end_token = min(start_token + chunk_size, len(matches))
        char_start = matches[start_token].start()
        char_end = matches[end_token - 1].end()
        chunks.append(
            {
                "chunk_index": chunk_index,
                "token_start": start_token,
                "token_end": end_token,
                "char_start": char_start,
                "char_end": char_end,
                "text": text[char_start:char_end],
            }
        )
        if end_token == len(matches):
            break
        start_token += step
        chunk_index += 1
    return chunks


def build_records(source: Path, *, chunk_size: int = 400, overlap: int = 60) -> tuple[list[dict], list[dict]]:
    """Crea registros de documentos y fragmentos sin persistirlos aún."""
    root, paths = discover_documents(source)
    document_records: list[dict] = []
    chunk_records: list[dict] = []

    for path in paths:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"El documento no es UTF-8 válido: {path}") from exc
        if not content.strip():
            raise ValueError(f"El documento está vacío: {path}")

        relative_path = path.relative_to(root).as_posix()
        content_hash = sha256_text(content)
        document_id = make_document_id(relative_path, content_hash)
        pieces = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        document_records.append(
            {
                "document_id": document_id,
                "source_path": relative_path,
                "filename": path.name,
                "content_sha256": content_hash,
                "char_count": len(content),
                "chunk_count": len(pieces),
            }
        )
        for piece in pieces:
            index = int(piece["chunk_index"])
            chunk_records.append(
                {
                    "chunk_id": make_chunk_id(
                        document_id, index, int(piece["char_start"]), int(piece["char_end"])
                    ),
                    "document_id": document_id,
                    "source_path": relative_path,
                    "filename": path.name,
                    "chunk_index": index,
                    "token_start": int(piece["token_start"]),
                    "token_end": int(piece["token_end"]),
                    "char_start": int(piece["char_start"]),
                    "char_end": int(piece["char_end"]),
                    "text": str(piece["text"]),
                    "content_sha256": content_hash,
                }
            )
    return document_records, chunk_records
