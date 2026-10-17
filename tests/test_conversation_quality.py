import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

import conversation_quality
from conversation_quality import (
    CIVIL,
    DERAILED,
    REPLY_SCORE_COLUMNS,
    TEXT_FEATURE_COLUMNS,
    UNCIVIL,
    VOTE_FEATURES,
    add_reply_scores,
    context_utterances,
    conversation_example,
    feature_model,
    load_replies,
    majority_baseline,
    paired_accuracy,
    structural_columns,
    text_model,
)
from evaluation import evaluate_classifier
from schema import Conversation, Speaker, Utterance
from structural_features import LEAKY_FEATURES

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def thread(cid, texts, derailed, pair_id, split="train"):
    """A linear CGA-style chain; the last text plays the outcome comment."""
    utterances = [
        Utterance(
            id=f"{cid}:u{i}",
            source_id=f"u{i}",
            conversation_id=cid,
            parent_id=None if i == 0 else f"{cid}:u{i - 1}",
            speaker_id=f"s{i % 2}",
            text=text,
            text_clean=text,
            created_at=BASE + timedelta(minutes=10 * i),
            depth=i,
            score=1,
        )
        for i, text in enumerate(texts)
    ]
    return Conversation(
        id=cid,
        source_id=cid.rsplit(":", 1)[-1],
        source="reddit",
        created_at=BASE,
        utterances=utterances,
        speakers={s: Speaker(id=s, handle=s, source="reddit") for s in ("s0", "s1")},
        metadata={"source_metadata": {
            "has_removed_comment": derailed,
            "pair_id": pair_id,
            "split": split,
        }},
    )


# Toy corpus where the context words separate the classes perfectly.
TOY = pd.DataFrame([
    conversation_example(thread(f"x:c{i}", ["opening point", reply, "final"], derailed, f"c{i + 1}"))
    for i, (reply, derailed) in enumerate([
        ("thanks good point", False),
        ("you idiot liar", True),
        ("thanks fair point", False),
        ("you stupid liar", True),
    ])
])


class ContextTests(unittest.TestCase):
    def test_final_utterance_is_never_in_context(self):
        conv = thread("x:a", ["root", "reply", "REMOVED ATTACK"], True, "b")
        self.assertEqual([u.text for u in context_utterances(conv)], ["root", "reply"])

        row = conversation_example(conv)
        self.assertNotIn("REMOVED ATTACK", row["context_text"])
        self.assertEqual(row["n_context"], 2)
        self.assertEqual(row["n_utterances"], 2)  # structural features use the same cutoff

    def test_labels_and_split_come_from_source_metadata(self):
        row = conversation_example(thread("x:a", ["r", "s"], False, "b", split="test"))
        self.assertEqual((row["label"], row["split"], row["pair_id"]), (CIVIL, "test", "b"))
        self.assertEqual(conversation_example(thread("x:a", ["r", "s"], True, "b"))["label"], DERAILED)

    def test_structural_columns_never_include_reply_scores(self):
        frame = TOY.assign(**{c: [0.1, 0.2, 0.3, 0.4] for c in REPLY_SCORE_COLUMNS})
        self.assertFalse(set(REPLY_SCORE_COLUMNS) & set(structural_columns(frame, include_votes=True)))

    def test_structural_columns_exclude_leaky_and_optionally_votes(self):
        frame = TOY.assign(min_score=[1, -2, 3, 0])
        with_votes = structural_columns(frame, include_votes=True)
        without = structural_columns(frame, include_votes=False)
        self.assertFalse(set(LEAKY_FEATURES) & set(with_votes))
        self.assertIn("min_score", with_votes)
        self.assertFalse(set(VOTE_FEATURES) & set(without))
        self.assertNotIn("label", with_votes)


class ModelContractTests(unittest.TestCase):
    def test_majority_baseline(self):
        model = majority_baseline().fit(TOY, [CIVIL, CIVIL, CIVIL, DERAILED])
        self.assertEqual(model.predict(TOY), [CIVIL] * 4)
        self.assertEqual(model.quality_scores(TOY), [0.75] * 4)

    def test_text_model_satisfies_text_classifier(self):
        model = text_model().fit(TOY.context_text, TOY.label)
        texts = ["opening point\n\nthanks good point"]
        prediction = model.predict(texts)
        self.assertEqual(len(prediction), 1)  # handles a single input
        self.assertIsInstance(prediction[0], str)
        self.assertEqual(model.predict(texts), prediction)  # deterministic
        self.assertEqual(len(model.predict(list(TOY.context_text))), len(TOY))

    def test_quality_scores_are_probabilities_of_civil(self):
        model = text_model().fit(TOY.context_text, TOY.label)
        scores = model.quality_scores(TOY.context_text)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        civil = [s for s, y in zip(scores, TOY.label) if y == CIVIL]
        derailed = [s for s, y in zip(scores, TOY.label) if y == DERAILED]
        self.assertGreater(min(civil), max(derailed))

    def test_feature_model_runs_with_missing_values(self):
        columns = ["n_utterances", "mean_reply_gap"]
        frame = TOY.copy()
        frame.loc[0, "mean_reply_gap"] = None
        model = feature_model(columns).fit(frame, frame.label)
        self.assertEqual(len(model.predict(frame)), len(frame))

    def test_evaluate_classifier_on_hand_picked_example(self):
        model = text_model().fit(TOY.context_text, TOY.label)
        result = evaluate_classifier(model, list(TOY.context_text), list(TOY.label))
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.labels, [CIVIL, DERAILED])


class PairedAccuracyTests(unittest.TestCase):
    def test_counts_wins_ties_and_losses(self):
        ids = ["x:a", "x:b", "x:c", "x:d", "x:e", "x:f"]
        pairs = ["b", "a", "d", "c", "f", "e"]
        labels = [CIVIL, DERAILED] * 3
        scores = [0.9, 0.1, 0.5, 0.5, 0.2, 0.8]  # win, tie, loss
        self.assertAlmostEqual(paired_accuracy(scores, labels, pairs, ids), 0.5)

    def test_ignores_pairs_split_across_sets(self):
        self.assertTrue(pd.isna(paired_accuracy([0.9], [CIVIL], ["missing"], ["x:a"])))


class ReplyLevelTests(unittest.TestCase):
    def write_corpus(self, conversations):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        with handle:
            for conv in conversations:
                handle.write(conv.model_dump_json() + "\n")
        return handle.name

    def test_load_replies_labels_final_reply_and_keeps_openings_unlabeled(self):
        path = self.write_corpus([
            thread("x:a", ["root a", "reply a", "you are an idiot"], True, "b"),
            thread("x:b", ["root b", "reply b", "fair point"], False, "a"),
        ])
        finals, openings = load_replies(path)

        self.assertEqual(dict(zip(finals.text, finals.label)), {"you are an idiot": UNCIVIL, "fair point": CIVIL})
        self.assertTrue(set(TEXT_FEATURE_COLUMNS) <= set(finals.columns))
        self.assertNotIn("you are an idiot", set(openings.text))
        self.assertEqual(sorted(openings[openings.conversation_id == "x:a"].text), ["reply a", "root a"])

    def test_deleted_final_reply_is_dropped(self):
        conv = thread("x:c", ["root c", "[deleted]"], True, "d")
        conv.utterances[-1].is_deleted = True
        finals, openings = load_replies(self.write_corpus([conv]))
        self.assertTrue(finals.empty)
        self.assertEqual(list(openings.text), ["root c"])

    def test_reply_scores_are_out_of_fold_for_train(self):
        """No train conversation is scored by a reply model that saw its own or its partner's answer."""
        ids, pairs, splits = [], [], []
        for i in range(7):  # 6 train pairs (enough for 5 folds) + 1 val pair
            a, b = f"x:p{i}a", f"x:p{i}b"
            ids += [a, b]
            pairs += [f"p{i}b", f"p{i}a"]
            splits += ["val" if i == 6 else "train"] * 2
        examples = pd.DataFrame({"conversation_id": ids, "pair_id": pairs, "split": splits})
        finals = examples.assign(
            label=[UNCIVIL, CIVIL] * 7,
            text=[f"final {c}" for c in ids],
        )
        openings = pd.DataFrame({"conversation_id": ids, "text": [f"opening {c}" for c in ids]})

        calls = []

        class Recorder:
            def fit(self, texts, labels):
                self.seen = {t.removeprefix("final ") for t in texts}
                return self

            def quality_scores(self, texts):
                scored = {t.removeprefix("opening ") for t in texts}
                calls.append((self.seen, scored))
                return [0.5] * len(texts)

        original = conversation_quality.text_model
        conversation_quality.text_model = Recorder
        try:
            result = add_reply_scores(examples, finals, openings)
        finally:
            conversation_quality.text_model = original

        partner = {c: f"x:{p}" for c, p in zip(ids, pairs)}
        train = set(examples[examples.split == "train"].conversation_id)
        for seen, scored in calls:
            for cid in scored & train:
                self.assertNotIn(cid, seen)
                self.assertNotIn(partner[cid], seen)
        self.assertEqual(len(result), len(examples))
        self.assertTrue(result[list(REPLY_SCORE_COLUMNS)].notna().all().all())


if __name__ == "__main__":
    unittest.main()
