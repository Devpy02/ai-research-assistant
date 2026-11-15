
import unittest
from unittest.mock import patch

from ai_research_assistant.embeddings import (
    EmbeddingError,
    _validate_embeddings,
    embed_chunks,
    embed_texts,
    get_model_info,
)


class EmbeddingTests(unittest.TestCase):

    def test_valid_embeddings_are_accepted(self):
        vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        dimension = _validate_embeddings(
            vectors,
            expected_count=2,
            expected_dimension=3,
        )

        self.assertEqual(dimension, 3)

    def test_wrong_embedding_count_is_rejected(self):
        vectors = [[0.1, 0.2, 0.3]]

        with self.assertRaisesRegex(EmbeddingError, "esperaban 2"):
            _validate_embeddings(vectors, expected_count=2)

    def test_inconsistent_dimensions_are_rejected(self):
        vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5],
        ]

        with self.assertRaisesRegex(EmbeddingError, "Dimensión inconsistente"):
            _validate_embeddings(vectors, expected_count=2)

    def test_non_numeric_values_are_rejected(self):
        vectors = [[0.1, "invalid", 0.3]]

        with self.assertRaisesRegex(EmbeddingError, "no numérico"):
            _validate_embeddings(vectors, expected_count=1)

    def test_non_finite_values_are_rejected(self):
        vectors = [[0.1, float("inf"), 0.3]]

        with self.assertRaisesRegex(EmbeddingError, "no finito"):
            _validate_embeddings(vectors, expected_count=1)

    def test_empty_embeddings_are_rejected(self):
        with self.assertRaisesRegex(EmbeddingError, "lista de embeddings"):
            _validate_embeddings(None, expected_count=1)

    @patch("ai_research_assistant.embeddings._post_json")
    def test_unexpected_model_is_rejected(self, mock_post):
        mock_post.return_value = {
            "model": "otro-modelo",
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        with self.assertRaisesRegex(EmbeddingError, "Modelo inesperado"):
            embed_texts(["Texto de prueba"])

    @patch("ai_research_assistant.embeddings._post_json")
    def test_embed_texts_returns_valid_vectors(self, mock_post):
        mock_post.return_value = {
            "model": "nomic-embed-text:latest",
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
        }

        vectors = embed_texts(["Texto uno", "Texto dos"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 3)

    def test_duplicate_chunk_ids_are_rejected(self):
        chunks = [
            {"chunk_id": "chunk-1", "text": "Texto uno"},
            {"chunk_id": "chunk-1", "text": "Texto dos"},
        ]

        with self.assertRaisesRegex(EmbeddingError, "duplicado"):
            embed_chunks(chunks)

    def test_empty_chunk_text_is_rejected(self):
        chunks = [
            {"chunk_id": "chunk-1", "text": "   "},
        ]

        with self.assertRaisesRegex(EmbeddingError, "Texto vacío"):
            embed_chunks(chunks)

    @patch("ai_research_assistant.embeddings.embed_texts")
    def test_embed_chunks_preserves_chunk_ids(self, mock_embed):
        mock_embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        chunks = [
            {"chunk_id": "chunk-1", "text": "Texto uno"},
            {"chunk_id": "chunk-2", "text": "Texto dos"},
        ]

        results = embed_chunks(chunks, batch_size=2)

        self.assertEqual(
            [item["chunk_id"] for item in results],
            ["chunk-1", "chunk-2"],
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_embed.call_count, 1)

    @patch("ai_research_assistant.embeddings.embed_texts")
    def test_embed_chunks_processes_multiple_batches(self, mock_embed):
        mock_embed.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6]],
        ]

        chunks = [
            {"chunk_id": "chunk-1", "text": "Texto uno"},
            {"chunk_id": "chunk-2", "text": "Texto dos"},
            {"chunk_id": "chunk-3", "text": "Texto tres"},
        ]

        results = embed_chunks(chunks, batch_size=2)

        self.assertEqual(len(results), 3)
        self.assertEqual(mock_embed.call_count, 2)

    @patch("ai_research_assistant.embeddings._get_json")
    def test_model_info_contains_digest_and_dimension(self, mock_get):
        mock_get.return_value = {
            "models": [
                {
                    "name": "nomic-embed-text:latest",
                    "digest": "abc123",
                    "details": {
                        "family": "nomic-bert",
                        "embedding_length": 768,
                        "context_length": 2048,
                    },
                }
            ]
        }

        info = get_model_info()

        self.assertEqual(info["digest"], "abc123")
        self.assertEqual(info["embedding_length"], 768)
        self.assertEqual(info["context_length"], 2048)


if __name__ == "__main__":
    unittest.main()
