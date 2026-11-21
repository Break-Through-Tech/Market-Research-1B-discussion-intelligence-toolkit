import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

from quality_labeler import DIMENSIONS, SAME_AS_START  # noqa: E402
from quality_scorer import (  # noqa: E402
    OVERALL_DIMENSIONS,
    QualityScorer,
    evaluate_scores,
    large_errors,
    score_comments,
)

RUDE = ["you are an idiot", "what a stupid moron you are", "shut up idiot", "you clown, stupid take"]
GOOD = [
    "I see your point, but the data shows rents rose because supply fell",
    "Good question. The study found a small effect, so I would change the claim slightly",
    "Consider the counterexample of Norway, which taxes more and still grew",
    "Thanks, that is fair. One source is the 2019 census report on housing",
]
NO_CLAIM = ["lol", "ok", "this", "same"]


def training_frame() -> pd.DataFrame:
    records = []
    for i, text in enumerate(RUDE * 3):
        records.append({"reply": text, "civility": "0", "relevance": "1", "reasoning": "0", "constructiveness": "0"})
    for i, text in enumerate(GOOD * 3):
        records.append({"reply": text, "civility": "2", "relevance": "2", "reasoning": "2", "constructiveness": "2"})
    for i, text in enumerate(NO_CLAIM * 3):
        records.append({"reply": text, "civility": "2", "relevance": "1", "reasoning": "N/A", "constructiveness": "1"})
    frame = pd.DataFrame(records)
    frame["utterance_id"] = [f"u{i}" for i in range(len(frame))]
    frame["parent"] = SAME_AS_START
    frame["thread_start"] = "CMV: housing policy and taxes matter for growth"
    return frame


class QualityScorerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = training_frame()
        cls.scorer = QualityScorer(alpha=0.5, min_df=1).fit(cls.frame)

    def test_score_columns_and_range(self):
        scores = self.scorer.score(self.frame)
        expected = {f"{d}_score" for d in DIMENSIONS} | {"reasoning_na_prob", "overall_score"}
        self.assertEqual(set(scores.columns), expected)
        self.assertEqual(len(scores), len(self.frame))
        for dim in DIMENSIONS:
            self.assertTrue(scores[f"{dim}_score"].between(0, 2).all())
        self.assertTrue(scores["reasoning_na_prob"].between(0, 1).all())

    def test_learns_civility_direction(self):
        scores = score_comments(["you stupid idiot", "I see your point, the data shows otherwise"], scorer=self.scorer)
        self.assertLess(scores.loc[0, "civility_score"], scores.loc[1, "civility_score"])

    def test_na_probability_higher_for_no_claim(self):
        scores = score_comments(["lol", "the data shows rents rose because supply fell"], scorer=self.scorer)
        self.assertGreater(scores.loc[0, "reasoning_na_prob"], scores.loc[1, "reasoning_na_prob"])

    def test_overall_is_mean_of_selected_dimensions(self):
        scores = self.scorer.score(self.frame.head(3))
        expected = scores[[f"{d}_score" for d in OVERALL_DIMENSIONS]].mean(axis=1)
        np.testing.assert_allclose(scores["overall_score"], expected)

    def test_reply_only_input_works(self):
        scores = score_comments(["just a reply"], scorer=self.scorer)
        self.assertEqual(len(scores), 1)

    def test_context_is_used_when_given(self):
        with_ctx = score_comments(["housing taxes"], scorer=self.scorer, parents=["housing taxes"],
                                  thread_starts=["housing taxes"])
        without = score_comments(["housing taxes"], scorer=self.scorer)
        self.assertNotEqual(with_ctx.loc[0, "relevance_score"], without.loc[0, "relevance_score"])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.scorer.save(Path(tmp) / "nested" / "scorer.joblib")
            loaded = QualityScorer.load(path)
        pd.testing.assert_frame_equal(loaded.score(self.frame), self.scorer.score(self.frame))

    def test_unfitted_scorer_raises(self):
        with self.assertRaises(RuntimeError):
            QualityScorer().score(pd.DataFrame({"reply": ["x"]}))

    def test_all_na_reasoning_raises(self):
        frame = self.frame.copy()
        frame["reasoning"] = "N/A"
        with self.assertRaises(ValueError):  # no numeric reasoning labels to regress on
            QualityScorer(min_df=1).fit(frame)

    def test_no_na_reasoning_gives_zero_probability(self):
        frame = self.frame[self.frame["reasoning"] != "N/A"]
        scorer = QualityScorer(min_df=1).fit(frame)
        self.assertTrue((scorer.score(frame)["reasoning_na_prob"] == 0).all())


class EvaluateScoresTest(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({
            "utterance_id": ["a", "b", "c", "d"],
            "civility": ["0", "1", "2", "2"],
            "relevance": ["2", "2", "2", "1"],
            "reasoning": ["0", "N/A", "2", "1"],
            "constructiveness": ["0", "1", "2", "1"],
        })

    def test_perfect_scores(self):
        predicted = self.reference.copy()  # labels as columns, like LLM output
        table = evaluate_scores(self.reference, predicted).set_index("dimension")
        self.assertTrue((table["mae"] == 0).all())
        self.assertAlmostEqual(table.loc["civility", "spearman"], 1.0)
        self.assertEqual(table.loc["reasoning", "n"], 3)  # N/A reference left out
        self.assertEqual(table.loc["reasoning", "na_accuracy"], 1.0)

    def test_float_scores(self):
        predicted = pd.DataFrame({
            "utterance_id": ["a", "b", "c", "d"],
            **{f"{d}_score": [0.5, 1.0, 1.5, 1.5] for d in DIMENSIONS},
            "reasoning_na_prob": [0.1, 0.9, 0.1, 0.2],
        })
        table = evaluate_scores(self.reference, predicted).set_index("dimension")
        self.assertAlmostEqual(table.loc["civility", "mae"], (0.5 + 0 + 0.5 + 0.5) / 4)
        self.assertAlmostEqual(table.loc["civility", "mae_predict_mean"], (1.25 + 0.25 + 0.75 + 0.75) / 4)
        self.assertEqual(table.loc["reasoning", "na_accuracy"], 1.0)

    def test_large_errors(self):
        predicted = pd.DataFrame({
            "utterance_id": ["a", "b", "c", "d"],
            **{f"{d}_score": [2.0, 1.0, 2.0, 1.0] for d in DIMENSIONS},
            "reasoning_na_prob": 0.0,
        })
        errors = large_errors(self.reference, predicted)
        self.assertIn("a", set(errors["utterance_id"]))  # civility 0 scored 2.0
        self.assertIn("civility_error", errors.columns)


if __name__ == "__main__":
    unittest.main()
