"""Offline regression tests for baseline eligibility, splits, and evaluation guards."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts"))

from discourse_data import load_discourse_data, save_json, split_manifest
from standardize_convokit import canonical, normalize

CORPUS = "reddit-coarse-discourse-corpus"


def conversation(cid):
    """Each thread contains both labels, so small dev/test fixtures stay valid."""
    def row(uid, parent, text, label):
        return {"id": uid, "root": cid, "reply-to": parent, "user": "alice",
                "text": text, "timestamp": None, "meta": {"majority_type": label}}

    rows = [
        row(cid, None, "  Why &amp; how?  ", "question"),
        row(f"{cid}-answer", cid, "Because.", "answer"),
        row(f"{cid}-deleted", cid, "[deleted]", "answer"),
        row(f"{cid}-other", cid, "Other text", "other"),
        row(f"{cid}-unlabeled", cid, "No annotation", None),
        row(f"{cid}-empty", cid, " \t ", "answer"),
        row(f"{cid}-orphan", f"{cid}-missing", "Orphan answer", "answer"),
    ]
    conv, _ = normalize(CORPUS, cid, rows, {}, {"alice": {}})
    return conv


def all_examples(splits):
    return splits.train + splits.dev + splits.test


class DiscourseDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "corpus.jsonl"
        self.conversations = [conversation(f"thread-{i}") for i in range(60)]
        self.write(self.conversations)

    def write(self, conversations):
        self.path.write_text("".join(c.model_dump_json() + "\n" for c in conversations),
                             encoding="utf-8")

    def test_baseline_filtering_uses_clean_text_and_retains_deleted_markers(self):
        examples = all_examples(load_discourse_data(self.path))
        expected = {
            canonical(CORPUS, "utterance", f"thread-{i}{suffix}")
            for i in range(60) for suffix in ("", "-answer", "-deleted", "-orphan")
        }
        self.assertEqual({ex.id for ex in examples}, expected)
        root = next(ex for ex in examples if ex.id == canonical(CORPUS, "utterance", "thread-0"))
        self.assertEqual(root.text, "Why & how?")
        self.assertEqual(sum(ex.text == "[deleted]" for ex in examples), 60)

    def test_synthetic_nodes_are_excluded_even_with_nonempty_labeled_text(self):
        for conv in self.conversations:
            ghost = next(u for u in conv.utterances if u.metadata.get("is_synthetic"))
            ghost.text_clean = "This is a structural placeholder"
            ghost.discourse_act = "answer"
        self.write(self.conversations)
        examples = all_examples(load_discourse_data(self.path))
        self.assertEqual(len(examples), 240)
        self.assertFalse(any(ex.text == "This is a structural placeholder" for ex in examples))
        self.assertTrue(all(ex.metadata["parent_text"] == "" for ex in examples
                            if ex.text == "Orphan answer"))

    def test_parent_text_is_available_even_if_parent_is_not_a_training_example(self):
        for conv in self.conversations:
            conv.root().discourse_act = "other"
            # Keep question represented in every conversation's retained labels.
            next(u for u in conv.utterances if u.text_clean == "Orphan answer").discourse_act = "question"
        self.write(self.conversations)
        examples = all_examples(load_discourse_data(self.path))
        answers = [ex for ex in examples if ex.text == "Because."]
        self.assertEqual(len(answers), 60)
        self.assertTrue(all(ex.metadata["parent_text"] == "Why & how?" for ex in answers))

    def test_splits_are_repeatable_order_independent_and_conversation_disjoint(self):
        first = load_discourse_data(self.path)
        groups = [{ex.metadata["conversation_id"] for ex in getattr(first, name)}
                  for name in ("train", "dev", "test")]
        self.assertTrue(all(groups))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        # Freeze the existing baseline's seed-42 hash-bucket assignments.
        self.assertEqual(tuple(len(getattr(first, name)) for name in ("train", "dev", "test")),
                         (164, 24, 52))
        self.write(list(reversed(self.conversations)))
        self.assertEqual(split_manifest(first), split_manifest(load_discourse_data(self.path)))

    def test_manifest_detects_text_label_parent_and_split_changes(self):
        original = load_discourse_data(self.path)
        manifest = split_manifest(original)
        for field in ("text", "label", "parent_text", "split"):
            with self.subTest(field=field):
                changed = load_discourse_data(self.path)
                example = changed.train[0]
                if field == "parent_text":
                    example.metadata[field] += " changed"
                elif field == "split":
                    changed.dev.append(changed.train.pop(0))
                else:
                    setattr(example, field, getattr(example, field) + " changed")
                self.assertNotEqual(manifest, split_manifest(changed))

    def test_duplicate_examples_and_wrong_corpus_are_rejected(self):
        self.write(self.conversations + [self.conversations[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate example ID"):
            load_discourse_data(self.path)
        wrong = self.conversations[0].model_copy(deep=True)
        wrong.id = "reddit:conversations-gone-awry-cmv-corpus:conversation:thread-0"
        for u in wrong.utterances:
            u.conversation_id = wrong.id
        self.write([wrong])
        with self.assertRaisesRegex(ValueError, "Coarse Discourse"):
            load_discourse_data(self.path)

    def test_evaluation_rejects_manifest_mismatch_before_loading_checkpoint(self):
        spec = importlib.util.spec_from_file_location(
            "evaluate_discourse_models_under_test", ROOT / "scripts/evaluate_discourse_models.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        splits = load_discourse_data(self.path)
        manifest = split_manifest(splits)
        manifest["examples"][0]["sha256"] = "modified text fingerprint"
        run = Path(self.temp.name) / "run"
        save_json(run / "split_manifest.json", manifest)
        output = Path(self.temp.name) / "evaluation"
        argv = ["evaluate", "--data", str(self.path), "--run", str(run), "--output", str(output)]
        stderr = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                cli.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Data or split differs", stderr.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
