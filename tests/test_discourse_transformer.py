"""Tiny local checkpoints exercise training/inference without a GPU or network.

Install torch>=2.3 and transformers>=5 to run these integration checks; the
remaining suite can run without those optional training dependencies.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

from corpora import ClassificationExample, ClassificationSplits
from discourse_data import save_json
from discourse_transformer import DiscourseTransformer, train_transformer
from evaluation import classification_metrics

HAS_TRAINING_DEPS = all(importlib.util.find_spec(name) is not None
                        for name in ("torch", "transformers"))


@unittest.skipUnless(HAS_TRAINING_DEPS, "Requires optional torch and transformers dependencies")
class DiscourseTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer

        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name) / "base"
        cls.base.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "yes", "why", "hello"]
        (cls.base / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        tokenizer = DistilBertTokenizer(vocab={token: i for i, token in enumerate(vocab)})
        tokenizer.save_pretrained(cls.base)
        # Deliberately reverse alphabetical order to detect incorrect label decoding.
        model = DistilBertForSequenceClassification(DistilBertConfig(
            vocab_size=len(vocab), dim=16, hidden_dim=32, n_layers=1, n_heads=2,
            num_labels=2, id2label={0: "question", 1: "answer"},
            label2id={"question": 0, "answer": 1}))
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([0., 5.]))
        model.save_pretrained(cls.base)
        save_json(cls.base / "discourse_settings.json", {"max_length": 16, "parent_context": False})

    def test_reload_preserves_label_mapping_and_batch_predictions(self):
        model = DiscourseTransformer(self.base, device="cpu")
        self.assertEqual(model.predict(["why", "yes", "hello"], batch_size=2), ["answer"] * 3)
        self.assertEqual(model.predict([]), [])
        self.assertEqual(model.model.config.id2label, {0: "question", 1: "answer"})

    def test_context_requires_parents_and_reserves_child_token_budget(self):
        model = DiscourseTransformer(self.base, device="cpu")
        model.settings["parent_context"] = True
        with self.assertRaisesRegex(ValueError, "requires parent texts"):
            model.predict(["yes"])
        encoded = model._encode(["yes " * 100], ["hello " * 100])
        tokens = model.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        self.assertLessEqual(len(tokens), 16)
        self.assertEqual(tokens.count("yes"), 6)
        self.assertEqual(tokens.count("hello"), 6)
        self.assertEqual(tokens.count("[SEP]"), 2)
        examples = [ClassificationExample(id="root", text="yes", label="answer",
                                          metadata={"parent_text": ""})]
        self.assertEqual(model.predict_examples(examples), ["answer"])

    def test_weighted_training_checkpoint_round_trip_in_both_context_modes(self):
        train = [ClassificationExample(
            id=f"train-{i}", text="why" if i == 0 else "yes",
            label="question" if i == 0 else "answer",
            metadata={"conversation_id": "train", "parent_text": "hello"}) for i in range(5)]
        dev = [ClassificationExample(id=f"dev-{i}", text=text, label=label,
                                     metadata={"conversation_id": "dev", "parent_text": ""})
               for i, (text, label) in enumerate((("why", "question"), ("yes", "answer")))]
        # Training must never consult test labels/examples.
        class UnavailableTest:
            def __iter__(self):
                raise AssertionError("Training accessed the held-out test split")

        splits = ClassificationSplits(train=train, dev=dev, test=UnavailableTest())
        with tempfile.TemporaryDirectory() as temp, patch.object(self.torch.cuda, "is_available", return_value=False):
            for context in (False, True):
                with self.subTest(parent_context=context):
                    output = Path(temp) / str(context)
                    with patch.object(self.torch.nn, "CrossEntropyLoss",
                                      wraps=self.torch.nn.CrossEntropyLoss) as loss_factory:
                        checkpoint = train_transformer(
                            splits, output, model_name=str(self.base), epochs=2, batch_size=2,
                            max_length=16, weighted=True, parent_context=context,
                            accumulation_steps=2, mixed_precision=False)
                    self.torch.testing.assert_close(
                        loss_factory.call_args.kwargs["weight"], self.torch.tensor([0.625, 2.5]))
                    reloaded = DiscourseTransformer(checkpoint, device="cpu")
                    result = classification_metrics([ex.label for ex in dev], reloaded.predict_examples(dev))
                    history = json.loads((output / "history.json").read_text())
                    compute = json.loads((output / "compute.json").read_text())
                    best = max(row["dev"]["macro_f1"] for row in history)
                    self.assertAlmostEqual(result.macro_f1, best)
                    self.assertAlmostEqual(compute["best_dev_macro_f1"], best)
                    self.assertEqual(len(history), 2)
                    self.assertTrue(reloaded.settings["weighted"])
                    self.assertEqual(reloaded.settings["parent_context"], context)
                    self.assertEqual(reloaded.model.config.label2id, {"answer": 0, "question": 1})
                    self.assertEqual(result.n_examples, 2)
                    self.assertGreater(reloaded.model.classifier.weight.abs().sum().item(), 0)
                    self.assertIsNone(compute["peak_gpu_memory_bytes"])
                    with self.assertRaisesRegex(ValueError, "Output must be empty"):
                        train_transformer(splits, output, model_name=str(self.base))


if __name__ == "__main__":
    unittest.main()
