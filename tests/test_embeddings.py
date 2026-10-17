import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'artifacts'))
from schema import Conversation, Speaker, Utterance
from embeddings import (
    build_text_table, embed_and_cache, load_cache, model_slug,
    sample_conversations, save_cache,
)


class FakeEmbedder:
    """Deterministic stand-in so tests never download a model."""
    model_name = "fake/model"
    max_seq_length = 128
    model = None

    def embed(self, texts, show_progress_bar=False):
        vectors = np.array([[len(t), t.count(" "), 1.0] for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_utt(uid, text, parent=None, cid="c1", **extra):
    return Utterance(
        id=uid, source_id=uid, conversation_id=cid, speaker_id="s1", parent_id=parent,
        text=text, text_clean=text, created_at=None, depth=0 if parent is None else 1, **extra
    )


def make_conv(cid, utterances):
    return Conversation(
        id=cid, source_id=cid, source="reddit", created_at=None,
        speakers={"s1": Speaker(id="s1", handle="s1", source="reddit")}, utterances=utterances,
    )


def sample_conversation():
    return make_conv("c1", [
        make_utt("1", "What is this?", discourse_act="question"),
        make_utt("ghost", "", parent="1", metadata={"is_synthetic": True}),
        make_utt("2", "It is a test.", parent="ghost", discourse_act="answer"),
        make_utt("3", "", parent="1"),
        make_utt("4", "[deleted]", parent="1", is_deleted=True),
    ])


class TextTableTests(unittest.TestCase):
    def test_excludes_synthetic_placeholders(self):
        table = build_text_table([sample_conversation()])
        self.assertEqual(list(table["utterance_id"]), ["1", "2", "3", "4"])

    def test_flags_empty_and_deleted(self):
        table = build_text_table([sample_conversation()]).set_index("utterance_id")
        self.assertTrue(table.loc["3", "is_empty"])
        self.assertFalse(table.loc["1", "is_empty"])
        self.assertTrue(table.loc["4", "is_deleted"])
        self.assertEqual(table.loc["2", "discourse_act"], "answer")

    def test_limit(self):
        self.assertEqual(len(build_text_table([sample_conversation()], limit=2)), 2)

    def test_empty_input_keeps_columns(self):
        table = build_text_table([])
        self.assertEqual(len(table), 0)
        self.assertIn("utterance_id", table.columns)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.convs = [make_conv(f"c{i}", [make_utt(f"u{i}", "hi", cid=f"c{i}")]) for i in range(200)]

    def test_deterministic(self):
        first = [c.id for c in sample_conversations(self.convs, 0.3)]
        second = [c.id for c in sample_conversations(self.convs, 0.3)]
        self.assertEqual(first, second)
        self.assertTrue(20 < len(first) < 100)

    def test_fraction_bounds(self):
        self.assertEqual(len(list(sample_conversations(self.convs, 0.0))), 0)
        self.assertEqual(len(list(sample_conversations(self.convs, 1.0))), 200)

    def test_custom_key_keeps_groups_together(self):
        kept = {c.id for c in sample_conversations(self.convs, 0.5, key=lambda c: "same-pair")}
        self.assertIn(len(kept), (0, 200))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.table = build_text_table([sample_conversation()])

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_keeps_alignment(self):
        out = embed_and_cache("toy", FakeEmbedder(), root=self.root, table=self.table,
                              extra_manifest={"sample": "all"})
        self.assertEqual(out, self.root / "fake__model" / "toy")
        cached = load_cache(out)
        self.assertEqual(list(cached.ids["utterance_id"]), list(self.table["utterance_id"]))
        np.testing.assert_allclose(cached.embeddings, FakeEmbedder().embed(self.table["text"]))
        self.assertEqual(cached.manifest["n_rows"], 4)
        self.assertEqual(cached.manifest["dimension"], 3)
        self.assertEqual(cached.manifest["sample"], "all")
        self.assertNotIn("text", cached.ids.columns)

    def test_float16(self):
        vectors = FakeEmbedder().embed(self.table["text"])
        save_cache(self.root, self.table, vectors, {}, dtype="float16")
        self.assertEqual(load_cache(self.root).embeddings.dtype, np.float16)

    def test_missing_manifest_is_refused(self):
        vectors = FakeEmbedder().embed(self.table["text"])
        save_cache(self.root, self.table, vectors, {})
        (self.root / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_cache(self.root)

    def test_row_mismatch_is_detected(self):
        vectors = FakeEmbedder().embed(self.table["text"])
        save_cache(self.root, self.table, vectors, {})
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        manifest["n_rows"] = 99
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_cache(self.root)

    def test_save_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            save_cache(self.root, self.table, np.zeros((2, 3), dtype=np.float32), {})

    def test_model_slug(self):
        self.assertEqual(model_slug("BAAI/bge-small-en-v1.5"), "BAAI__bge-small-en-v1.5")


if __name__ == "__main__":
    unittest.main()
