"""Synthetic export fixtures verify pairing and artifact audits, without models."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

from compare_parent_context import (
    align_evaluations, compare, load_evaluation, reddit_url, sample_changes,
)
from evaluation import classification_metrics


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plain = self.root / "plain_test"
        self.context = self.root / "parent_context_test"
        self.output = self.root / "comparison"
        # All four correctness transitions, including an empty-parent example.
        self.rows = [
            {"id": f"reddit:reddit-coarse-discourse-corpus:utterance:c{i}",
             "conversation_id": "reddit:reddit-coarse-discourse-corpus:conversation:abc",
             "text": f"reply {i}", "parent_text": "" if i == 3 else "parent text",
             "gold": "question" if i in (0, 2) else "answer"}
            for i in range(4)
        ]
        self.export(self.plain, ["answer", "answer", "question", "question"])
        self.export(self.context, ["question", "question", "question", "question"], reverse=True)

    def export(self, folder, predictions, reverse=False):
        (folder / "transformer").mkdir(parents=True)
        rows = [{**row, "prediction": prediction} for row, prediction in zip(self.rows, predictions)]
        write_json(folder / "evaluation_config.json", {
            "split": "test", "split_seed": 42, "n_examples": 4, "run": str(folder),
        })
        # Build the hash independently, using the upstream manifest format.
        manifest = {"version": 1, "seed": 42, "dev_frac": .1, "test_frac": .1,
                    "labels": ["answer", "question"], "examples": [
                        {"id": row["id"], "conversation_id": row["conversation_id"], "split": "test",
                         "sha256": hashlib.sha256(json.dumps(
                             [row["text"], row["gold"], row["parent_text"]],
                             ensure_ascii=False).encode()).hexdigest()} for row in rows]}
        write_json(folder / "split_manifest.json", manifest)
        self.write_predictions(folder, list(reversed(rows)) if reverse else rows)
        write_json(folder / "transformer" / "metrics.json", classification_metrics(
            [row["gold"] for row in rows], predictions).model_dump())

    def write_predictions(self, folder, rows):
        (folder / "transformer" / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def predictions(self, folder):
        return [json.loads(line) for line in (folder / "transformer" / "predictions.jsonl").read_text().splitlines()]

    def test_end_to_end_pairs_by_id_and_exports_complete_report(self):
        summary = compare(self.plain, self.context, self.output)
        self.assertEqual(summary["n_examples"], 4)
        self.assertEqual(summary["transitions"]["all"], {
            "n_examples": 4, "helps": 1, "hurts": 1, "both_correct": 1, "both_wrong": 1,
            "comment_only_accuracy": .5, "parent_context_accuracy": .5})
        self.assertEqual(summary["transitions"]["available"]["n_examples"], 3)
        self.assertEqual(summary["transitions"]["empty"]["both_wrong"], 1)
        self.assertEqual(summary["metrics"]["comment_only"]["macro_f1"], .5)
        self.assertAlmostEqual(summary["metrics"]["parent_context"]["macro_f1"], 1 / 3)
        self.assertTrue((self.output / "per_class_f1.csv").is_file())
        report = (self.output / "report.md").read_text()
        self.assertIn("Context helps", report)
        self.assertIn("Context hurts", report)
        self.assertIn("https://www.reddit.com/comments/abc/_/c0/", report)
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), summary)
        self.assertFalse((self.output / "predictions.jsonl").exists())

    def test_rejects_duplicate_missing_or_wrong_ids(self):
        original = self.predictions(self.plain)
        for rows in (original + [original[0]], original[:-1], [{**original[0], "id": "unknown"}, *original[1:]]):
            with self.subTest(n=len(rows)):
                self.write_predictions(self.plain, rows)
                with self.assertRaises(ValueError):
                    load_evaluation(self.plain)

    def test_manifest_fingerprint_catches_label_parent_and_reply_changes(self):
        original = self.predictions(self.plain)
        for field, value in (("gold", "answer"), ("parent_text", "different context"), ("text", "different reply")):
            changed = [dict(row) for row in original]
            changed[0][field] = value
            self.write_predictions(self.plain, changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "manifest"):
                load_evaluation(self.plain)

    def test_rejects_different_splits_and_manifests(self):
        plain, context = load_evaluation(self.plain), load_evaluation(self.context)
        context["config"]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "different split"):
            align_evaluations(plain, context)
        context["config"]["split"] = "test"
        context["manifest"]["examples"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different split manifests"):
            align_evaluations(plain, context)

    def test_rejects_conversation_leakage_and_duplicate_manifest_ids(self):
        location = self.plain / "split_manifest.json"
        manifest = json.loads(location.read_text())
        manifest["examples"][1]["split"] = "train"
        write_json(location, manifest)
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            load_evaluation(self.plain)
        manifest["examples"][1]["split"] = "test"
        manifest["examples"].append(dict(manifest["examples"][0]))
        write_json(location, manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate manifest"):
            load_evaluation(self.plain)

    def test_rejects_stale_metrics_and_unknown_prediction_labels(self):
        location = self.plain / "transformer" / "metrics.json"
        metrics = json.loads(location.read_text())
        metrics["macro_f1"] = .999
        write_json(location, metrics)
        with self.assertRaisesRegex(ValueError, "macro_f1"):
            load_evaluation(self.plain)
        predictions = self.predictions(self.plain)
        predictions[0]["prediction"] = "unknown"
        self.write_predictions(self.plain, predictions)
        with self.assertRaisesRegex(ValueError, "Invalid prediction"):
            load_evaluation(self.plain)

    def test_refuses_to_overwrite_an_existing_report(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            compare(self.plain, self.context, self.output)
        self.assertEqual(marker.read_text(), "preserve me")

    def test_missing_inputs_fail_before_creating_output(self):
        with self.assertRaises(FileNotFoundError):
            compare(self.root / "missing", self.context, self.output)
        self.assertFalse(self.output.exists())

    def test_sampling_repeatable_and_source_links_are_bounded(self):
        rows = align_evaluations(load_evaluation(self.plain), load_evaluation(self.context))
        self.assertEqual(sample_changes(rows), sample_changes(rows))
        self.assertEqual({row["effect"] for row in sample_changes(rows)}, {"helps", "hurts"})
        self.assertEqual(sample_changes(rows, 0), [])
        self.assertIsNone(reddit_url({"id": "unrelated:id", "conversation_id": "unrelated:thread"}))
        self.assertIsNone(reddit_url({"id": "reddit:reddit-coarse-discourse-corpus:utterance:../../bad",
                                     "conversation_id": self.rows[0]["conversation_id"]}))


if __name__ == "__main__":
    unittest.main()
