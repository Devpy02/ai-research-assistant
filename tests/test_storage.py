import json
import tempfile
import unittest
from pathlib import Path

from ai_research_assistant.storage import load_index, persist_documents, verify_sources


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = self.root / "documents"
        self.docs.mkdir()
        self.index = self.root / "index"
        self.source = self.docs / "manual.md"
        self.content = "# Manual\n\nOllama expone una API local.\nEl modelo genera respuestas.\nLos embeddings representan textos.\n"
        self.source.write_text(self.content, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_persist_and_reload_preserves_content_and_provenance(self):
        manifest = persist_documents(self.docs, self.index, chunk_size=5, overlap=1)
        loaded_manifest, chunks = load_index(self.index)
        self.assertEqual(manifest["chunk_count"], len(chunks))
        self.assertEqual(loaded_manifest["document_count"], 1)
        self.assertTrue(all(row["source_path"] == "manual.md" for row in chunks))
        self.assertTrue(all(row["document_id"] == manifest["documents"][0]["document_id"] for row in chunks))
        self.assertTrue(all(row["text"] for row in chunks))
        self.assertEqual(len({row["chunk_id"] for row in chunks}), len(chunks))
        self.assertEqual(verify_sources(self.index, self.docs), [])

    def test_ids_stable_across_reindex_of_same_content(self):
        persist_documents(self.docs, self.index, chunk_size=5, overlap=1)
        _, first = load_index(self.index)
        first_ids = [row["chunk_id"] for row in first]
        persist_documents(self.docs, self.index, chunk_size=5, overlap=1)
        _, second = load_index(self.index)
        self.assertEqual(first_ids, [row["chunk_id"] for row in second])

    def test_source_modification_is_detected(self):
        persist_documents(self.docs, self.index, chunk_size=5, overlap=1)
        self.source.write_text(self.content + "Nueva información.\n", encoding="utf-8")
        errors = verify_sources(self.index, self.docs)
        self.assertTrue(any("contenido cambió" in error for error in errors))

    def test_modified_jsonl_is_detected_by_hash(self):
        persist_documents(self.docs, self.index, chunk_size=5, overlap=1)
        path = self.index / "chunks.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash de chunks.jsonl"):
            load_index(self.index)

    def test_empty_document_is_rejected(self):
        self.source.write_text("   \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "vacío"):
            persist_documents(self.docs, self.index, chunk_size=5, overlap=1)


if __name__ == "__main__":
    unittest.main()
