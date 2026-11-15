
import unittest

import numpy as np
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ai_research_assistant.vector_storage import (
    persist_vector_index,
    load_vector_index,
)
from ai_research_assistant.vector_storage import (
    VectorStorageError,
    build_vector_matrix,
)


class TestBuildVectorMatrix(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {"chunk_id": "chunk_001"},
            {"chunk_id": "chunk_002"},
        ]

        self.embeddings = [
            {"chunk_id": "chunk_001", "embedding": [0.1, 0.2, 0.3]},
            {"chunk_id": "chunk_002", "embedding": [0.4, 0.5, 0.6]},
        ]

    def test_builds_valid_float32_matrix(self):
        matrix, chunk_ids = build_vector_matrix(
            self.embeddings,
            self.chunks,
            expected_dimension=3,
        )

        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(
            chunk_ids,
            ["chunk_001", "chunk_002"],
        )

    def test_rejects_empty_embeddings(self):
        with self.assertRaises(VectorStorageError):
            build_vector_matrix([], self.chunks)

    def test_rejects_empty_chunks(self):
        with self.assertRaises(VectorStorageError):
            build_vector_matrix(self.embeddings, [])

    def test_rejects_duplicate_chunk_ids(self):
        embeddings = [
            {"chunk_id": "chunk_001", "embedding": [0.1, 0.2]},
            {"chunk_id": "chunk_001", "embedding": [0.3, 0.4]},
        ]

        with self.assertRaises(VectorStorageError):
            build_vector_matrix(
                embeddings,
                self.chunks,
            )

    def test_rejects_different_chunk_order(self):
        embeddings = [
            {"chunk_id": "chunk_002", "embedding": [0.1, 0.2, 0.3]},
            {"chunk_id": "chunk_001", "embedding": [0.4, 0.5, 0.6]},
        ]

        with self.assertRaises(VectorStorageError):
            build_vector_matrix(embeddings, self.chunks)

    def test_rejects_incorrect_dimension(self):
        embeddings = [
            {"chunk_id": "chunk_001", "embedding": [0.1, 0.2]},
            {"chunk_id": "chunk_002", "embedding": [0.3, 0.4]},
        ]

        with self.assertRaises(VectorStorageError):
            build_vector_matrix(
                embeddings,
                self.chunks,
                expected_dimension=3,
            )

    def test_rejects_non_numeric_values(self):
        embeddings = [
            {"chunk_id": "chunk_001", "embedding": [0.1, "invalid"]},
            {"chunk_id": "chunk_002", "embedding": [0.3, 0.4]},
        ]

        with self.assertRaises(VectorStorageError):
            build_vector_matrix(embeddings, self.chunks)

    def test_rejects_non_finite_values(self):
        embeddings = [
            {"chunk_id": "chunk_001", "embedding": [0.1, float("nan")]},
            {"chunk_id": "chunk_002", "embedding": [0.3, 0.4]},
        ]

        with self.assertRaises(VectorStorageError):
            build_vector_matrix(embeddings, self.chunks)

    def test_rejects_boolean_values(self):
        embeddings = [
            {"chunk_id": "chunk_001", "embedding": [True, 0.2]},
            {"chunk_id": "chunk_002", "embedding": [0.3, 0.4]},
        ]

        with self.assertRaises(VectorStorageError):
            build_vector_matrix(embeddings, self.chunks)


class TestVectorIndexPersistence(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "vector_index"

        self.matrix = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        )

        self.chunk_ids = ["chunk_001", "chunk_002"]

        self.model_info = {
            "name": "nomic-embed-text:latest",
            "digest": "a" * 64,
            "family": "nomic-bert",
            "format": "gguf",
            "parameter_size": "137M",
            "quantization_level": "F16",
            "embedding_length": 3,
            "context_length": 2048,
        }

        self.manifest_hash = "b" * 64
        self.chunks_hash = "c" * 64

        self.chunks = [
            {"chunk_id": "chunk_001"},
            {"chunk_id": "chunk_002"},
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rejects_non_dictionary_model_info(self):
        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                None,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )

    def test_rejects_incomplete_model_info(self):
        invalid_model_info = [
            {},
            {"name": "nomic-embed-text:latest"},
            {
                "name": "nomic-embed-text:latest",
                "digest": "a" * 64,
            },
        ]

        for model_info in invalid_model_info:
            with self.subTest(model_info=model_info):
                with self.assertRaises(VectorStorageError):
                    load_vector_index(
                        self.output_dir,
                        self.chunks,
                        model_info,
                        source_manifest_sha256=self.manifest_hash,
                        source_chunks_sha256=self.chunks_hash,
                    )


    def test_persists_and_loads_vector_index(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        manifest, loaded_matrix = load_vector_index(
            self.output_dir,
            self.chunks,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        np.testing.assert_array_equal(loaded_matrix, self.matrix)
        self.assertEqual(manifest["chunk_ids"], self.chunk_ids)
        self.assertEqual(manifest["vector_count"], 2)
        self.assertEqual(manifest["dimension"], 3)

    def test_creates_current_pointer(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        current = (self.output_dir / "CURRENT").read_text(
            encoding="utf-8"
        ).strip()

        self.assertEqual(current, "version_001")

    def test_detects_modified_vectors_file(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        current = (self.output_dir / "CURRENT").read_text(
            encoding="utf-8"
        ).strip()

        vectors_path = (
            self.output_dir / "versions" / current / "vectors.npy"
        )

        with vectors_path.open("ab") as handle:
            handle.write(b"modified")

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                self.model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )

    def test_rejects_changed_model_digest(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        changed_model = self.model_info.copy()
        changed_model["digest"] = "d" * 64

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                changed_model,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )

    def test_rejects_changed_document_index(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                self.model_info,
                source_manifest_sha256="e" * 64,
                source_chunks_sha256=self.chunks_hash,
            )

    def test_preserves_previous_version_when_publishing_new_one(self):
        # Publicar la primera versión.
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        # Publicar una segunda versión.
        second_matrix = self.matrix + np.float32(0.1)

        persist_vector_index(
            self.output_dir,
            second_matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        versions_dir = self.output_dir / "versions"

        # Ambas versiones deben conservarse.
        first_version = versions_dir / "version_001"
        second_version = versions_dir / "version_002"

        self.assertTrue(first_version.is_dir())
        self.assertTrue(second_version.is_dir())

        # CURRENT debe apuntar a la segunda versión.
        current = (self.output_dir / "CURRENT").read_text(
            encoding="utf-8"
        ).strip()

        self.assertEqual(current, "version_002")

        # La versión activa debe contener la segunda matriz.
        _, loaded_matrix = load_vector_index(
            self.output_dir,
            self.chunks,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        np.testing.assert_array_equal(loaded_matrix, second_matrix)

        # La primera versión debe conservar sus vectores originales.
        first_matrix = np.load(
            first_version / "vectors.npy",
            allow_pickle=False,
        )

        np.testing.assert_array_equal(first_matrix, self.matrix)

    def test_failed_publication_preserves_previous_version(self):
        # Crear y activar la primera versión.
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        current_before = (
            self.output_dir / "CURRENT"
        ).read_text(encoding="utf-8").strip()

        # Preparar una segunda matriz.
        second_matrix = self.matrix + np.float32(0.1)

        # Simular un fallo al escribir el manifiesto.
        with patch(
            "ai_research_assistant.vector_storage._atomic_write",
            side_effect=OSError("Error simulado de escritura"),
        ):
            with self.assertRaises(OSError):
                persist_vector_index(
                    self.output_dir,
                    second_matrix,
                    self.chunk_ids,
                    self.model_info,
                    source_manifest_sha256=self.manifest_hash,
                    source_chunks_sha256=self.chunks_hash,
                )

        # CURRENT debe continuar apuntando a la primera versión.
        current_after = (
            self.output_dir / "CURRENT"
        ).read_text(encoding="utf-8").strip()

        self.assertEqual(current_before, "version_001")
        self.assertEqual(current_after, "version_001")

        # La primera versión debe seguir siendo recuperable.
        _, loaded_matrix = load_vector_index(
            self.output_dir,
            self.chunks,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        np.testing.assert_array_equal(loaded_matrix, self.matrix)

    def test_rejects_invalid_current_pointer(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        current_path = self.output_dir / "CURRENT"

        # Simular una referencia manipulada.
        current_path.write_text(
            "../../other_directory",
            encoding="utf-8",
        )

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                self.model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )


    def test_rejects_boolean_vector_count_in_manifest(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        current = (self.output_dir / "CURRENT").read_text(
            encoding="utf-8"
        ).strip()

        manifest_path = (
            self.output_dir / "versions" / current / "manifest.json"
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # JSON permite true, pero no debe aceptarse como cantidad de vectores.
        manifest["vector_count"] = True

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                self.model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )
    def test_rejects_invalid_chunk_ids_when_persisting(self):
        invalid_ids = ["chunk_001", ""]

        with self.assertRaises(VectorStorageError):
            persist_vector_index(
                self.output_dir,
                self.matrix,
                invalid_ids,
                self.model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )


    def test_rejects_non_list_chunk_ids_when_persisting(self):
        with self.assertRaises(VectorStorageError):
            persist_vector_index(
                self.output_dir,
                self.matrix,
                None,
                self.model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )
    def test_rejects_invalid_source_index_in_manifest(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        current = (self.output_dir / "CURRENT").read_text(
            encoding="utf-8"
        ).strip()

        manifest_path = (
            self.output_dir / "versions" / current / "manifest.json"
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_index"] = []

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                self.model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )


    def test_rejects_invalid_model_section_in_manifest(self):
        persist_vector_index(
            self.output_dir,
            self.matrix,
            self.chunk_ids,
            self.model_info,
            source_manifest_sha256=self.manifest_hash,
            source_chunks_sha256=self.chunks_hash,
        )

        current = (self.output_dir / "CURRENT").read_text(
            encoding="utf-8"
        ).strip()

        manifest_path = (
            self.output_dir / "versions" / current / "manifest.json"
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model"] = []

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                self.model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )

    def test_rejects_invalid_model_name(self):
        invalid_model_info = self.model_info.copy()
        invalid_model_info["name"] = ""

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                invalid_model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )

    def test_rejects_invalid_model_digest(self):
        invalid_model_info = self.model_info.copy()
        invalid_model_info["digest"] = "invalid-digest"

        with self.assertRaises(VectorStorageError):
            load_vector_index(
                self.output_dir,
                self.chunks,
                invalid_model_info,
                source_manifest_sha256=self.manifest_hash,
                source_chunks_sha256=self.chunks_hash,
            )

    def test_rejects_invalid_embedding_dimension(self):
        invalid_dimensions = [0, -1, True, 3.0, "768", None]

        for dimension in invalid_dimensions:
            invalid_model_info = self.model_info.copy()
            invalid_model_info["embedding_length"] = dimension

            with self.subTest(dimension=dimension):
                with self.assertRaises(VectorStorageError):
                    load_vector_index(
                        self.output_dir,
                        self.chunks,
                        invalid_model_info,
                        source_manifest_sha256=self.manifest_hash,
                        source_chunks_sha256=self.chunks_hash,
                    )


if __name__ == "__main__":
    unittest.main()
