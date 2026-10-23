import unittest

from ai_research_assistant.chunking import chunk_text, make_chunk_id, make_document_id


class ChunkingTests(unittest.TestCase):
    def test_overlap_and_exact_character_offsets(self):
        text = "uno dos tres cuatro cinco seis"
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        self.assertEqual([row["text"] for row in chunks], [
            "uno dos tres", "tres cuatro cinco", "cinco seis"
        ])
        for row in chunks:
            self.assertEqual(text[row["char_start"]:row["char_end"]], row["text"])

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(chunk_text(" \n\t ", chunk_size=3, overlap=1), [])

    def test_invalid_chunk_configuration_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("texto", chunk_size=0, overlap=0)
        with self.assertRaises(ValueError):
            chunk_text("texto", chunk_size=3, overlap=3)

    def test_ids_are_deterministic_and_content_sensitive(self):
        doc_a = make_document_id("manual.md", "abc")
        doc_a_again = make_document_id("manual.md", "abc")
        doc_b = make_document_id("manual.md", "def")
        self.assertEqual(doc_a, doc_a_again)
        self.assertNotEqual(doc_a, doc_b)
        self.assertEqual(make_chunk_id(doc_a, 0, 0, 10), make_chunk_id(doc_a, 0, 0, 10))
        self.assertNotEqual(make_chunk_id(doc_a, 0, 0, 10), make_chunk_id(doc_a, 1, 0, 10))


if __name__ == "__main__":
    unittest.main()
