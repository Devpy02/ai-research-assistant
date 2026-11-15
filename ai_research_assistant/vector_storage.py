"""Persistencia NumPy de embeddings con validación de integridad y trazabilidad."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
VECTORS_FILENAME = "vectors.npy"
MANIFEST_FILENAME = "manifest.json"

CURRENT_FILENAME = "CURRENT"
VERSIONS_DIRNAME = "versions"
VERSION_PREFIX = "version_"

class VectorStorageError(ValueError):
    """Error de persistencia o validación del índice vectorial."""


def _sha256_file(path: Path) -> str:
    """Calcula SHA-256 leyendo el archivo por bloques."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Escribe un archivo temporal y lo sustituye de forma atómica."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

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


def build_vector_matrix(
    embeddings: list[dict],
    chunks: list[dict],
    *,
    expected_dimension: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Construye una matriz float32 y conserva la correspondencia con los chunks."""

    if not isinstance(embeddings, list) or not embeddings:
        raise VectorStorageError("La lista de embeddings está vacía o es inválida")

    if not isinstance(chunks, list) or not chunks:
        raise VectorStorageError("La lista de fragmentos está vacía o es inválida")

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]

    if any(not isinstance(value, str) or not value.strip() for value in chunk_ids):
        raise VectorStorageError("Hay fragmentos con chunk_id ausente o inválido")

    if len(chunk_ids) != len(set(chunk_ids)):
        raise VectorStorageError("Hay chunk_id duplicados en los fragmentos")

    embedding_ids = [item.get("chunk_id") for item in embeddings]

    if any(not isinstance(value, str) or not value.strip() for value in embedding_ids):
        raise VectorStorageError("Hay embeddings con chunk_id ausente o inválido")

    if len(embedding_ids) != len(set(embedding_ids)):
        raise VectorStorageError("Hay chunk_id duplicados en los embeddings")

    if embedding_ids != chunk_ids:
        raise VectorStorageError(
            "Los embeddings no corresponden a los fragmentos "
            "o su orden es diferente"
        )

    vectors = []

    for index, item in enumerate(embeddings):
        vector = item.get("embedding")

        if not isinstance(vector, list) or not vector:
            raise VectorStorageError(
                f"Embedding vacío o inválido en la posición {index}"
            )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in vector
        ):
            raise VectorStorageError(
                f"Valores no numéricos en el embedding {index}"
            )

        if expected_dimension is not None and len(vector) != expected_dimension:
            raise VectorStorageError(
                f"Dimensión incorrecta en embedding {index}: "
                f"{len(vector)}; esperada: {expected_dimension}"
            )

        vectors.append(vector)

    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VectorStorageError("No se pudo construir la matriz NumPy") from exc

    if matrix.ndim != 2:
        raise VectorStorageError("La matriz de embeddings debe tener dos dimensiones")

    if not np.isfinite(matrix).all():
        raise VectorStorageError("La matriz contiene valores no finitos")

    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise VectorStorageError("La dimensión final de la matriz es incorrecta")

    return matrix, chunk_ids



def build_vector_manifest(
    matrix: np.ndarray,
    chunk_ids: list[str],
    model_info: dict,
    *,
    vectors_sha256: str,
    source_manifest_sha256: str,
    source_chunks_sha256: str,
) -> dict:
    """Construye el manifiesto que describe y vincula el índice vectorial."""

    # Validar matriz.
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise VectorStorageError(
            "La matriz vectorial debe ser un ndarray bidimensional"
        )

    if matrix.dtype != np.float32:
        raise VectorStorageError(
            "La matriz vectorial debe utilizar float32"
        )

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise VectorStorageError("La matriz vectorial no puede estar vacía")

    if not np.isfinite(matrix).all():
        raise VectorStorageError("La matriz contiene valores no finitos")

    # Validar identificadores de fragmentos.
    if not isinstance(chunk_ids, list) or not chunk_ids:
        raise VectorStorageError(
            "La lista de chunk_id está vacía o es inválida"
        )

    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip()
        for chunk_id in chunk_ids
    ):
        raise VectorStorageError("Hay chunk_id ausentes o inválidos")

    if len(chunk_ids) != matrix.shape[0]:
        raise VectorStorageError(
            "La cantidad de chunk_id no coincide con las filas de la matriz"
        )

    if len(chunk_ids) != len(set(chunk_ids)):
        raise VectorStorageError("Hay chunk_id duplicados")

    # Validar información del modelo.
    if not isinstance(model_info, dict):
        raise VectorStorageError("La información del modelo es inválida")

    model_name = model_info.get("name")
    model_digest = model_info.get("digest")
    embedding_length = model_info.get("embedding_length")

    if not isinstance(model_name, str) or not model_name.strip():
        raise VectorStorageError("El nombre del modelo es inválido")

    if (
        not isinstance(model_digest, str)
        or len(model_digest) != 64
        or any(char not in "0123456789abcdef" for char in model_digest.lower())
    ):
        raise VectorStorageError("El digest del modelo es inválido")

    if (
        isinstance(embedding_length, bool)
        or not isinstance(embedding_length, int)
        or embedding_length <= 0
    ):
        raise VectorStorageError("La dimensión declarada del modelo es inválida")

    if embedding_length != matrix.shape[1]:
        raise VectorStorageError(
            "La dimensión de la matriz no coincide con la del modelo"
        )

    # Validar hashes.
    hashes = {
        "vectors_sha256": vectors_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_chunks_sha256": source_chunks_sha256,
    }

    for field, value in hashes.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value.lower())
        ):
            raise VectorStorageError(f"Hash SHA-256 inválido: {field}")

    return {
        "schema_version": SCHEMA_VERSION,
        "storage_format": "numpy_npy",
        "vectors_file": VECTORS_FILENAME,
        "vectors_sha256": vectors_sha256.lower(),
        "vector_count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "dtype": "float32",
        "model": {
            "name": model_name,
            "digest": model_digest.lower(),
            "family": model_info.get("family"),
            "format": model_info.get("format"),
            "parameter_size": model_info.get("parameter_size"),
            "quantization_level": model_info.get("quantization_level"),
            "embedding_length": embedding_length,
            "context_length": model_info.get("context_length"),
        },
        "source_index": {
            "manifest_sha256": source_manifest_sha256.lower(),
            "chunks_sha256": source_chunks_sha256.lower(),
        },
        "chunk_ids": chunk_ids.copy(),
    }


def persist_vector_index(
    output_dir: Path,
    matrix: np.ndarray,
    chunk_ids: list[str],
    model_info: dict,
    *,
    source_manifest_sha256: str,
    source_chunks_sha256: str,
) -> dict:
    """Construye, valida y publica una nueva versión del índice vectorial."""

    output_dir = Path(output_dir).resolve()
    versions_dir = output_dir / VERSIONS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    versions_dir.mkdir(parents=True, exist_ok=True)

    # Validar la matriz antes de comenzar la escritura.
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise VectorStorageError("La matriz debe ser bidimensional")

    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise VectorStorageError("La matriz debe contener valores float32 finitos")

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise VectorStorageError("La matriz no puede estar vacía")

    if not isinstance(chunk_ids, list) or not chunk_ids:
        raise VectorStorageError("La lista de chunk_id está vacía o es inválida")

    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip()
        for chunk_id in chunk_ids
    ):
        raise VectorStorageError("Hay chunk_id ausentes o inválidos")

    if len(chunk_ids) != matrix.shape[0]:
        raise VectorStorageError("La cantidad de IDs no coincide con las filas")

    if len(chunk_ids) != len(set(chunk_ids)):
        raise VectorStorageError("Hay chunk_id duplicados")

    # Determinar el siguiente número de versión.
    existing_versions = []

    for path in versions_dir.iterdir():
        if path.is_dir() and path.name.startswith(VERSION_PREFIX):
            suffix = path.name[len(VERSION_PREFIX):]
            if suffix.isdigit():
                existing_versions.append(int(suffix))

    next_version = max(existing_versions, default=0) + 1
    version_name = f"{VERSION_PREFIX}{next_version:03d}"

    final_dir = versions_dir / version_name

    if final_dir.exists():
        raise VectorStorageError(f"La versión ya existe: {version_name}")

    # Construir en un directorio temporal dentro de versions/.
    temp_dir = Path(
        tempfile.mkdtemp(prefix=".vector_tmp_", dir=versions_dir)
    )

    try:
        vectors_path = temp_dir / VECTORS_FILENAME

        with vectors_path.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

        vectors_sha256 = _sha256_file(vectors_path)

        manifest = build_vector_manifest(
            matrix,
            chunk_ids,
            model_info,
            vectors_sha256=vectors_sha256,
            source_manifest_sha256=source_manifest_sha256,
            source_chunks_sha256=source_chunks_sha256,
        )

        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n"
        ).encode("utf-8")

        _atomic_write(temp_dir / MANIFEST_FILENAME, manifest_bytes)

        # Comprobar la matriz guardada antes de publicar.
        loaded = np.load(vectors_path, allow_pickle=False)

        if loaded.shape != matrix.shape or not np.array_equal(loaded, matrix):
            raise VectorStorageError("La matriz guardada no coincide con la original")

        if _sha256_file(vectors_path) != manifest["vectors_sha256"]:
            raise VectorStorageError("El hash de la matriz guardada no coincide")

        # Publicar la versión completa.
        os.replace(temp_dir, final_dir)

        # Activar la nueva versión mediante un único archivo de referencia.
        _atomic_write(
            output_dir / CURRENT_FILENAME,
            (version_name + "\n").encode("utf-8"),
        )

        return manifest

    except Exception:
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_vector_index(
    output_dir: Path,
    chunks: list[dict],
    model_info: dict,
    *,
    source_manifest_sha256: str,
    source_chunks_sha256: str,
) -> tuple[dict, np.ndarray]:
    """Carga la versión activa y valida su integridad y correspondencia documental."""
   
    if not isinstance(chunks, list) or not chunks:
        raise VectorStorageError(
            "La lista de fragmentos está vacía o es inválida"
        )

    if any(not isinstance(chunk, dict) for chunk in chunks):
        raise VectorStorageError(
            "La lista contiene fragmentos con estructura inválida"
        )

    current_chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    
    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip()
        for chunk_id in current_chunk_ids
    ):
        raise VectorStorageError(
            "Hay fragmentos con chunk_id ausente o inválido"
        )

    if len(current_chunk_ids) != len(set(current_chunk_ids)):
        raise VectorStorageError(
            "Hay chunk_id duplicados en los fragmentos actuales"
        )

    if not isinstance(model_info, dict):
        raise VectorStorageError(
            "La información del modelo debe ser un diccionario"
        )

    required_model_fields = ("name", "digest", "embedding_length")

    if any(field not in model_info for field in required_model_fields):
        raise VectorStorageError(
            "La información del modelo está incompleta"
        )

    if not isinstance(model_info["name"], str) or not model_info["name"].strip():
        raise VectorStorageError("El nombre del modelo es inválido")

    if (
        not isinstance(model_info["digest"], str)
        or len(model_info["digest"]) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in model_info["digest"])
    ):
        raise VectorStorageError("El digest del modelo es inválido")

    if (
        isinstance(model_info["embedding_length"], bool)
        or not isinstance(model_info["embedding_length"], int)
        or model_info["embedding_length"] <= 0
    ):
        raise VectorStorageError("La dimensión del modelo es inválida")

    output_dir = Path(output_dir).resolve()
    current_path = output_dir / CURRENT_FILENAME

    if not current_path.is_file():
        raise FileNotFoundError(
            f"No existe el indicador de versión activa: {current_path}"
        )

    version_name = current_path.read_text(encoding="utf-8").strip()

    if (
        not version_name.startswith(VERSION_PREFIX)
        or not version_name[len(VERSION_PREFIX):].isdigit()
    ):
        raise VectorStorageError("El archivo CURRENT contiene una versión inválida")

    version_dir = output_dir / VERSIONS_DIRNAME / version_name

    if not version_dir.is_dir():
        raise VectorStorageError(f"No existe el directorio de versión: {version_name}")

    manifest_path = version_dir / MANIFEST_FILENAME

    if not manifest_path.is_file():
        raise VectorStorageError("No existe el manifiesto vectorial")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorStorageError("No se pudo leer el manifiesto vectorial") from exc

    if not isinstance(manifest, dict):
        raise VectorStorageError(
            "El manifiesto vectorial debe contener un objeto JSON"
        )

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise VectorStorageError("Versión de esquema vectorial no soportada")

    if manifest.get("storage_format") != "numpy_npy":
        raise VectorStorageError("Formato de almacenamiento vectorial no soportado")

    if manifest.get("vectors_file") != VECTORS_FILENAME:
        raise VectorStorageError("Nombre de archivo vectorial no válido")

    vectors_path = version_dir / VECTORS_FILENAME

    if not vectors_path.is_file():
        raise VectorStorageError("No existe el archivo vectors.npy")

    # Verificar la integridad física del archivo.
    if _sha256_file(vectors_path) != manifest.get("vectors_sha256"):
        raise VectorStorageError("El hash de vectors.npy no coincide con el manifiesto")

    # Verificar la correspondencia con el índice documental actual.
    source_index = manifest.get("source_index", {})

    if not isinstance(source_index, dict):
        raise VectorStorageError(
            "La sección source_index del manifiesto es inválida"
        )

    if source_index.get("manifest_sha256") != source_manifest_sha256:
        raise VectorStorageError("El manifiesto documental ha cambiado")

    if source_index.get("chunks_sha256") != source_chunks_sha256:
        raise VectorStorageError("El archivo de fragmentos ha cambiado")

    current_chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    stored_chunk_ids = manifest.get("chunk_ids")

    if not isinstance(stored_chunk_ids, list):
        raise VectorStorageError("La lista de chunk_id del manifiesto es inválida")

    if stored_chunk_ids != current_chunk_ids:
        raise VectorStorageError(
            "Los vectores no corresponden al índice documental actual "
            "o el orden de los fragmentos ha cambiado"
        )

    if len(stored_chunk_ids) != len(set(stored_chunk_ids)):
        raise VectorStorageError("El manifiesto contiene chunk_id duplicados")

    # Verificar identidad del modelo.
    stored_model = manifest.get("model", {})

    if not isinstance(stored_model, dict):
        raise VectorStorageError(
            "La sección model del manifiesto es inválida"
        )

    if stored_model.get("name") != model_info.get("name"):
        raise VectorStorageError("El nombre del modelo de embeddings ha cambiado")

    if stored_model.get("digest") != model_info.get("digest"):
        raise VectorStorageError("El digest del modelo de embeddings ha cambiado")

    if stored_model.get("embedding_length") != model_info.get("embedding_length"):
        raise VectorStorageError("La dimensión del modelo de embeddings ha cambiado")

    try:
        matrix = np.load(vectors_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise VectorStorageError("No se pudo cargar la matriz NumPy") from exc

    if matrix.ndim != 2:
        raise VectorStorageError("La matriz vectorial no tiene dos dimensiones")

    if matrix.dtype != np.float32:
        raise VectorStorageError("La matriz vectorial no utiliza float32")

    if not np.isfinite(matrix).all():
        raise VectorStorageError("La matriz contiene valores no finitos")

    expected_count = manifest.get("vector_count")
    expected_dimension = manifest.get("dimension")

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise VectorStorageError(
            "La cantidad de vectores declarada en el manifiesto es inválida"
        )

    if (
        isinstance(expected_dimension, bool)
        or not isinstance(expected_dimension, int)
        or expected_dimension <= 0
    ):
        raise VectorStorageError(
            "La dimensión declarada en el manifiesto es inválida"
        )

    if matrix.shape != (expected_count, expected_dimension):
        raise VectorStorageError(
            "Las dimensiones de la matriz no coinciden con el manifiesto"
        )
       
    if matrix.shape[0] != len(stored_chunk_ids):
        raise VectorStorageError("La cantidad de vectores no coincide con los chunk_id")

    if matrix.shape[1] != model_info["embedding_length"]:
        raise VectorStorageError("La dimensión de los vectores no coincide con el modelo")

    if manifest.get("dtype") != "float32":
        raise VectorStorageError("El tipo de datos declarado en el manifiesto es incorrecto")

    return manifest, matrix
