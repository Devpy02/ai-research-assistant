
import argparse
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_research_assistant.cli import make_parser, run_embedding, main
from ai_research_assistant.embeddings import EmbeddingError


class CLITests(unittest.TestCase):

    def test_parser_registers_all_commands(self):
        parser = make_parser()

        args = parser.parse_args([
            "embed",
            "--index", "data/index",
            "--output", "data/vector_index",
        ])

        self.assertEqual(args.command, "embed")
        self.assertEqual(args.model, "nomic-embed-text:latest")
        self.assertEqual(args.base_url, "http://localhost:11434")
        self.assertEqual(args.batch_size, 4)

    @patch("ai_research_assistant.cli.persist_vector_index")
    @patch("ai_research_assistant.cli.build_vector_matrix")
    @patch("ai_research_assistant.cli.embed_chunks")
    @patch("ai_research_assistant.cli.get_model_info")
    @patch("ai_research_assistant.cli.load_index")
    def test_run_embedding_executes_complete_pipeline(
        self,
        mock_load_index,
        mock_model_info,
        mock_embed_chunks,
        mock_build_matrix,
        mock_persist,
    ):
        index_dir = Path("data/index")
        output_dir = Path("data/vector_index")

        chunks = [{"chunk_id": "chk_001", "text": "Texto de prueba"}]
        document_manifest = {
            "document_count": 1,
            "chunks_sha256": "a" * 64,
        }
        model_info = {
            "name": "nomic-embed-text:latest",
            "digest": "b" * 64,
            "embedding_length": 768,
        }
        embeddings = [{"chunk_id": "chk_001", "embedding": [0.1] * 768}]
        matrix = [[0.1] * 768]
        chunk_ids = ["chk_001"]
        vector_manifest = {
            "vector_count": 1,
            "dimension": 768,
            "vectors_sha256": "c" * 64,
        }

        mock_load_index.return_value = (document_manifest, chunks)
        mock_model_info.return_value = model_info
        mock_embed_chunks.return_value = embeddings
        mock_build_matrix.return_value = (matrix, chunk_ids)
        mock_persist.return_value = vector_manifest

        with patch.object(Path, "read_bytes", return_value=b"manifest de prueba"):
            args = argparse.Namespace(
                index=index_dir,
                output=output_dir,
                model=model_info["name"],
                base_url="http://localhost:11434",
                batch_size=4,
            )

            result = run_embedding(args)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["document_count"], 1)
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["vector_count"], 1)
        self.assertEqual(result["dimension"], 768)

        mock_load_index.assert_called_once_with(index_dir)
        mock_model_info.assert_called_once()
        mock_embed_chunks.assert_called_once()
        mock_build_matrix.assert_called_once()
        mock_persist.assert_called_once()

    @patch("ai_research_assistant.cli.run_embedding")
    def test_main_handles_embedding_error(self, mock_run_embedding):
        mock_run_embedding.side_effect = EmbeddingError(
            "Ollama no está disponible"
        )

        with patch(
            "sys.argv",
            [
                "cli.py",
                "embed",
                "--index", "data/index",
                "--output", "data/vector_index",
            ],
        ):
            with patch("builtins.print") as mock_print:
                exit_code = main()

        self.assertEqual(exit_code, 2)
        mock_print.assert_called_once()
        self.assertIn("ERROR:", mock_print.call_args[0][0])

    def test_parser_preserves_existing_commands(self):
        cases = [
            ["ingest", "--source", "docs", "--output", "idx"],
            ["show", "--index", "idx"],
            ["verify", "--source", "docs", "--index", "idx"],
        ]

        for argv in cases:
            with self.subTest(argv=argv):
                args = make_parser().parse_args(argv)
                self.assertEqual(args.command, argv[0])


if __name__ == "__main__":
    unittest.main()
