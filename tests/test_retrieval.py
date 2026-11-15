
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import numpy as np

from ai_research_assistant.retrieval import search, RetrievalError


class TestRetrieval(unittest.TestCase):

    def setUp(self):
        self.index_dir = Path("data/index")
        self.vector_dir = Path("data/vector_index")

    def test_rejects_empty_query(self):
        with self.assertRaises(RetrievalError):
            search(
                "",
                index_dir=self.index_dir,
                vector_dir=self.vector_dir,
            )

    def test_rejects_whitespace_query(self):
        with self.assertRaises(RetrievalError):
            search(
                "   ",
                index_dir=self.index_dir,
                vector_dir=self.vector_dir,
            )

    def test_rejects_invalid_top_k(self):
        invalid_values = [0, -1, True, 1.5, "3"]

        for top_k in invalid_values:
            with self.subTest(top_k=top_k):
                with self.assertRaises(RetrievalError):
                    search(
                        "consulta válida",
                        index_dir=self.index_dir,
                        vector_dir=self.vector_dir,
                        top_k=top_k,
                    )

    def test_returns_results_in_similarity_order(self):
        chunks = [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "filename": "manual.md",
                "source_path": "manual.md",
                "chunk_index": 0,
                "text": "Primer fragmento",
            },
            {
                "chunk_id": "chunk_002",
                "document_id": "doc_001",
                "filename": "manual.md",
                "source_path": "manual.md",
                "chunk_index": 1,
                "text": "Segundo fragmento",
            },
            {
                "chunk_id": "chunk_003",
                "document_id": "doc_001",
                "filename": "manual.md",
                "source_path": "manual.md",
                "chunk_index": 2,
                "text": "Tercer fragmento",
            },
        ]

        matrix = np.array(
            [
                [0.6, 0.8],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        model_info = {
            "name": "nomic-embed-text:latest",
            "digest": "a" * 64,
            "embedding_length": 2,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            (index_dir / "manifest.json").write_bytes(b"{}")

            with (
                patch(
                    "ai_research_assistant.retrieval.load_index",
                    return_value=(
                        {"chunks_sha256": "b" * 64},
                        chunks,
                    ),
                ),
                patch(
                    "ai_research_assistant.retrieval.get_model_info",
                    return_value=model_info,
                ),
                patch(
                    "ai_research_assistant.retrieval.load_vector_index",
                    return_value=({}, matrix),
                ),
                patch(
                    "ai_research_assistant.retrieval.embed_texts",
                    return_value=[[1.0, 0.0]],
                ),
            ):
                results = search(
                    "consulta de prueba",
                    index_dir=index_dir,
                    vector_dir=Path(temp_dir) / "vectors",
                    top_k=3,
                )

        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["chunk_002", "chunk_001", "chunk_003"],
        )

        self.assertAlmostEqual(results[0]["score"], 1.0)
        self.assertAlmostEqual(results[1]["score"], 0.6)
        self.assertAlmostEqual(results[2]["score"], 0.0)

        self.assertEqual(results[0]["text"], "Segundo fragmento")
        self.assertEqual(results[0]["filename"], "manual.md")

    def test_respects_top_k(self):
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "document_id": "doc_001",
                "filename": "manual.md",
                "source_path": "manual.md",
                "chunk_index": i,
                "text": f"Fragmento {i}",
            }
            for i in range(3)
        ]

        matrix = np.array(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
            dtype=np.float32,
        )

        model_info = {
            "name": "nomic-embed-text:latest",
            "digest": "a" * 64,
            "embedding_length": 2,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            (index_dir / "manifest.json").write_bytes(b"{}")

            with (
                patch(
                    "ai_research_assistant.retrieval.load_index",
                    return_value=({"chunks_sha256": "b" * 64}, chunks),
                ),
                patch(
                    "ai_research_assistant.retrieval.get_model_info",
                    return_value=model_info,
                ),
                patch(
                    "ai_research_assistant.retrieval.load_vector_index",
                    return_value=({}, matrix),
                ),
                patch(
                    "ai_research_assistant.retrieval.embed_texts",
                    return_value=[[1.0, 0.0]],
                ),
            ):
                results = search(
                    "consulta",
                    index_dir=index_dir,
                    vector_dir=Path(temp_dir) / "vectors",
                    top_k=2,
                )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["chunk_0", "chunk_1"],
        )

    def test_rejects_zero_query_vector(self):
        chunks = [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "filename": "manual.md",
                "source_path": "manual.md",
                "chunk_index": 0,
                "text": "Fragmento de prueba",
            }
        ]

        matrix = np.array([[1.0, 0.0]], dtype=np.float32)

        model_info = {
            "name": "nomic-embed-text:latest",
            "digest": "a" * 64,
            "embedding_length": 2,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            (index_dir / "manifest.json").write_bytes(b"{}")

            with (
                patch(
                    "ai_research_assistant.retrieval.load_index",
                    return_value=({"chunks_sha256": "b" * 64}, chunks),
                ),
                patch(
                    "ai_research_assistant.retrieval.get_model_info",
                    return_value=model_info,
                ),
                patch(
                    "ai_research_assistant.retrieval.load_vector_index",
                    return_value=({}, matrix),
                ),
                patch(
                    "ai_research_assistant.retrieval.embed_texts",
                    return_value=[[0.0, 0.0]],
                ),
            ):
                with self.assertRaises(RetrievalError):
                    search(
                        "consulta",
                        index_dir=index_dir,
                        vector_dir=Path(temp_dir) / "vectors",
                    )
    def test_stops_if_document_index_cannot_be_loaded(self):
        with (
            patch(
                "ai_research_assistant.retrieval.load_index",
                side_effect=ValueError("Índice documental inválido"),
            ),
            patch(
                "ai_research_assistant.retrieval.get_model_info"
            ) as mock_model,
            patch(
                "ai_research_assistant.retrieval.load_vector_index"
            ) as mock_vectors,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Índice documental inválido",
            ):
                search(
                    "consulta válida",
                    index_dir=self.index_dir,
                    vector_dir=self.vector_dir,
                )

        mock_model.assert_not_called()
        mock_vectors.assert_not_called()
    def test_stops_if_model_info_cannot_be_loaded(self):
        chunks = [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "filename": "manual.md",
                "source_path": "manual.md",
                "chunk_index": 0,
                "text": "Fragmento de prueba",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            (index_dir / "manifest.json").write_bytes(b"{}")

            with (
                patch(
                    "ai_research_assistant.retrieval.load_index",
                    return_value=({"chunks_sha256": "b" * 64}, chunks),
                ),
                patch(
                    "ai_research_assistant.retrieval.get_model_info",
                    side_effect=ConnectionError("Ollama no disponible"),
                ),
                patch(
                    "ai_research_assistant.retrieval.load_vector_index"
                ) as mock_vectors,
                patch(
                    "ai_research_assistant.retrieval.embed_texts"
                ) as mock_embeddings,
            ):
                with self.assertRaisesRegex(
                    ConnectionError,
                    "Ollama no disponible",
                ):
                    search(
                        "consulta válida",
                        index_dir=index_dir,
                        vector_dir=Path(temp_dir) / "vectors",
                    )

            mock_vectors.assert_not_called()
            mock_embeddings.assert_not_called()
    def test_rejects_zero_vector_in_matrix(self):
        chunks = [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "filename": "manual.md",
                "source_path": "manual.md",
                "chunk_index": 0,
                "text": "Fragmento de prueba",
            }
        ]

        matrix = np.array([[0.0, 0.0]], dtype=np.float32)

        model_info = {
            "name": "nomic-embed-text:latest",
            "digest": "a" * 64,
            "embedding_length": 2,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            (index_dir / "manifest.json").write_bytes(b"{}")

            with (
                patch(
                    "ai_research_assistant.retrieval.load_index",
                    return_value=({"chunks_sha256": "b" * 64}, chunks),
                ),
                patch(
                    "ai_research_assistant.retrieval.get_model_info",
                    return_value=model_info,
                ),
                patch(
                    "ai_research_assistant.retrieval.load_vector_index",
                    return_value=({}, matrix),
                ),
                patch(
                    "ai_research_assistant.retrieval.embed_texts",
                    return_value=[[1.0, 0.0]],
                ),
            ):
                with self.assertRaisesRegex(
                    RetrievalError,
                    "vectores nulos",
                ):
                    search(
                        "consulta válida",
                        index_dir=index_dir,
                        vector_dir=Path(temp_dir) / "vectors",
                    )   
    