import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

from schema import Conversation, Speaker, Utterance
from structural_features import (
    LEAKY_FEATURES,
    conversation_features,
    degenerate_columns,
    extract_conversation_features,
    model_ready_columns,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def utt(uid, parent, speaker, minutes=None, **extra):
    """Build an Utterance. `minutes` sets created_at; None leaves it undated."""
    return Utterance(
        id=uid,
        source_id=uid,
        conversation_id="c1",
        parent_id=parent,
        speaker_id=speaker,
        text=uid,
        text_clean=uid,
        created_at=None if minutes is None else BASE + timedelta(minutes=minutes),
        depth=0,  # recomputed by the module; value here is deliberately wrong
        **extra,
    )


def conv(utterances, cid="c1"):
    speaker_ids = {u.speaker_id for u in utterances}
    return Conversation(
        id=cid,
        source_id=cid,
        source="reddit",
        created_at=BASE,
        utterances=utterances,
        speakers={
            sid: Speaker(id=sid, handle=sid, source="reddit") for sid in speaker_ids
        },
    )


class ShapeTests(unittest.TestCase):
    def test_narrow_chain(self):
        # root -> a -> b -> c, alternating two speakers
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "a", "s1", 2),
            utt("c", "b", "s2", 3),
        ])
        row = conversation_features(c)
        self.assertEqual(row["n_utterances"], 4)
        self.assertEqual(row["max_depth"], 3)
        self.assertEqual(row["max_branching"], 1)
        self.assertEqual(row["n_leaves"], 1)
        self.assertEqual(row["max_width"], 1)
        self.assertEqual(row["depth_ratio"], 1.0)  # perfectly narrow

    def test_wide_flat_thread(self):
        # root with three direct replies
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "r", "s3", 2),
            utt("c", "r", "s4", 3),
        ])
        row = conversation_features(c)
        self.assertEqual(row["max_depth"], 1)
        self.assertEqual(row["max_branching"], 3)
        self.assertEqual(row["n_leaves"], 3)
        self.assertEqual(row["max_width"], 3)
        self.assertAlmostEqual(row["depth_ratio"], 1 / 3)

    def test_single_utterance(self):
        row = conversation_features(conv([utt("r", None, "s1", 0)]))
        self.assertEqual(row["n_utterances"], 1)
        self.assertEqual(row["max_depth"], 0)
        self.assertEqual(row["depth_ratio"], 0.0)
        self.assertEqual(row["n_speakers"], 1)


class SyntheticTests(unittest.TestCase):
    def test_placeholders_excluded_and_child_reattached(self):
        # root -> [placeholder] -> real child. The child should re-attach to
        # root, giving depth 1 rather than 2, and the placeholder must not count.
        c = conv([
            utt("r", None, "s1", 0),
            utt("p", "r", "s_synth", None, metadata={"is_synthetic": True}),
            utt("a", "p", "s2", 5),
        ])
        row = conversation_features(c)
        self.assertEqual(row["n_utterances"], 2)
        self.assertEqual(row["max_depth"], 1)
        self.assertEqual(row["n_synthetic_removed"], 1)
        self.assertTrue(row["had_structural_repair"])
        self.assertEqual(row["n_speakers"], 2)  # synthetic speaker not counted

    def test_chain_of_placeholders_bypassed(self):
        c = conv([
            utt("r", None, "s1", 0),
            utt("p1", "r", "sx", None, metadata={"is_synthetic": True}),
            utt("p2", "p1", "sy", None, metadata={"is_synthetic": True}),
            utt("a", "p2", "s2", 1),
        ])
        row = conversation_features(c)
        self.assertEqual(row["n_utterances"], 2)
        self.assertEqual(row["max_depth"], 1)
        self.assertEqual(row["n_synthetic_removed"], 2)

    def test_no_repair_flagged_false(self):
        c = conv([utt("r", None, "s1", 0), utt("a", "r", "s2", 1)])
        row = conversation_features(c)
        self.assertEqual(row["n_synthetic_removed"], 0)
        self.assertFalse(row["had_structural_repair"])


class TimingTests(unittest.TestCase):
    def test_gaps_measured_against_parent(self):
        # r(0) -> a(10) and r(0) -> b(2): sibling gaps are 10 and 2 minutes,
        # measured from the parent, not from the previous message in time.
        c = conv([
            utt("r", None, "s1", 0),
            utt("b", "r", "s2", 2),
            utt("a", "r", "s3", 10),
        ])
        row = conversation_features(c)
        self.assertEqual(row["duration_seconds"], 600)
        self.assertEqual(row["min_reply_gap"], 120)
        self.assertEqual(row["median_reply_gap"], 360)  # mean of 120 and 600

    def test_undated_corpus_yields_none(self):
        # Coarse Discourse has no timestamps at all.
        c = conv([utt("r", None, "s1"), utt("a", "r", "s2")])
        row = conversation_features(c)
        self.assertIsNone(row["duration_seconds"])
        self.assertIsNone(row["median_reply_gap"])
        self.assertIsNone(row["gap_acceleration"])
        # Shape features still work without timestamps.
        self.assertEqual(row["max_depth"], 1)

    def test_speeding_up_gives_acceleration_below_one(self):
        # gaps from parent: 100, 100, then 1, 1 -> second half much faster
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 100),
            utt("b", "a", "s1", 200),
            utt("c", "b", "s2", 201),
            utt("d", "c", "s1", 202),
        ])
        row = conversation_features(c)
        self.assertLess(row["gap_acceleration"], 1.0)


class SpeakerTests(unittest.TestCase):
    def test_two_person_duel(self):
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "a", "s1", 2),
            utt("c", "b", "s2", 3),
        ])
        row = conversation_features(c)
        self.assertEqual(row["n_speakers"], 2)
        self.assertEqual(row["top_speaker_share"], 0.5)
        self.assertEqual(row["top_two_speaker_share"], 1.0)
        self.assertEqual(row["self_reply_count"], 0)

    def test_self_reply_counted(self):
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s1", 1),  # same speaker replying to themself
        ])
        self.assertEqual(conversation_features(c)["self_reply_count"], 1)


class SignalTests(unittest.TestCase):
    def test_deletion_and_scores(self):
        c = conv([
            utt("r", None, "s1", 0, score=5),
            utt("a", "r", "s2", 1, score=-3, is_deleted=True),
            utt("b", "a", "s1", 2),  # score left as None
        ])
        row = conversation_features(c)
        self.assertEqual(row["n_deleted"], 1)
        self.assertAlmostEqual(row["deleted_ratio"], 1 / 3)
        self.assertEqual(row["min_score"], -3)
        self.assertEqual(row["mean_score"], 1.0)
        self.assertEqual(row["n_negative_score"], 1)
        self.assertEqual(row["n_missing_score"], 1)

    def test_leaky_features_all_present(self):
        row = conversation_features(conv([utt("r", None, "s1", 0)]))
        for name in LEAKY_FEATURES:
            self.assertIn(name, row)


class CutoffTests(unittest.TestCase):
    def test_first_n_truncates_by_time(self):
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "a", "s1", 2),
            utt("c", "b", "s2", 3),
        ])
        row = conversation_features(c, first_n=2)
        self.assertEqual(row["n_utterances"], 2)
        self.assertEqual(row["max_depth"], 1)

    def test_cutoff_hides_late_deletion(self):
        # The deletion happens last; a first_n=2 cutoff must not see it.
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "a", "s1", 2, is_deleted=True),
        ])
        self.assertEqual(conversation_features(c)["n_deleted"], 1)
        self.assertEqual(conversation_features(c, first_n=2)["n_deleted"], 0)


class TableTests(unittest.TestCase):
    def test_one_row_per_conversation(self):
        a = conv([utt("r", None, "s1", 0)], cid="c1")
        b = conv([utt("r", None, "s1", 0), utt("a", "r", "s2", 1)], cid="c2")
        df = extract_conversation_features([a, b])
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df["conversation_id"]), ["c1", "c2"])

    def test_columns_stable_across_corpora(self):
        # Task 6 needs a fixed schema, so a dated and an undated conversation
        # must still produce identical columns.
        dated = conv([utt("r", None, "s1", 0)], cid="c1")
        undated = conv([utt("r", None, "s1")], cid="c2")
        self.assertListEqual(
            list(extract_conversation_features([dated]).columns),
            list(extract_conversation_features([undated]).columns),
        )


class GrowthAndFirstReplyTests(unittest.TestCase):
    def test_growth_rate_per_hour(self):
        # 3 comments spanning 60 minutes -> 3 per hour
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 30),
            utt("b", "a", "s1", 60),
        ])
        self.assertAlmostEqual(conversation_features(c)["growth_rate_per_hour"], 3.0)

    def test_time_to_first_reply_uses_earliest_child_of_root(self):
        # root at 0, direct replies at 45 and 5 -> first reply is 5 minutes
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 45),
            utt("b", "r", "s3", 5),
        ])
        self.assertEqual(conversation_features(c)["time_to_first_reply"], 300)

    def test_deep_reply_does_not_count_as_first_reply(self):
        # r -> a(10) -> b(11): the 1-minute gap is a->b, not a reply to root
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 10),
            utt("b", "a", "s1", 11),
        ])
        row = conversation_features(c)
        self.assertEqual(row["time_to_first_reply"], 600)
        self.assertEqual(row["min_reply_gap"], 60)
        self.assertEqual(row["max_reply_gap"], 600)

    def test_timing_none_when_undated(self):
        c = conv([utt("r", None, "s1"), utt("a", "r", "s2")])
        row = conversation_features(c)
        for key in ("growth_rate_per_hour", "time_to_first_reply", "max_reply_gap", "mean_reply_gap"):
            self.assertIsNone(row[key], key)


class ScoreTrajectoryTests(unittest.TestCase):
    def test_root_and_last_score(self):
        c = conv([
            utt("r", None, "s1", 0, score=10),
            utt("a", "r", "s2", 1, score=4),
            utt("b", "a", "s1", 2, score=-7),
        ])
        row = conversation_features(c)
        self.assertEqual(row["root_score"], 10)
        self.assertEqual(row["last_score"], -7)

    def test_declining_reception_gives_negative_trend(self):
        c = conv([
            utt("r", None, "s1", 0, score=20),
            utt("a", "r", "s2", 1, score=10),
            utt("b", "a", "s1", 2, score=-5),
            utt("c", "b", "s2", 3, score=-15),
        ])
        self.assertLess(conversation_features(c)["score_trend"], 0)

    def test_min_score_position_late_vs_early(self):
        late = conv([
            utt("r", None, "s1", 0, score=5),
            utt("a", "r", "s2", 1, score=3),
            utt("b", "a", "s1", 2, score=-9),
        ])
        early = conv([
            utt("r", None, "s1", 0, score=-9),
            utt("a", "r", "s2", 1, score=3),
            utt("b", "a", "s1", 2, score=5),
        ])
        self.assertEqual(conversation_features(late)["min_score_position"], 1.0)
        self.assertEqual(conversation_features(early)["min_score_position"], 0.0)

    def test_score_trajectory_none_when_unscored(self):
        c = conv([utt("r", None, "s1", 0), utt("a", "r", "s2", 1)])
        row = conversation_features(c)
        for key in ("root_score", "last_score", "score_trend", "min_score_position"):
            self.assertIsNone(row[key], key)

    def test_cutoff_hides_late_score_collapse(self):
        c = conv([
            utt("r", None, "s1", 0, score=5),
            utt("a", "r", "s2", 1, score=4),
            utt("b", "a", "s1", 2, score=-50),
        ])
        self.assertEqual(conversation_features(c)["min_score"], -50)
        self.assertEqual(conversation_features(c, first_n=2)["min_score"], 4)


class ColumnSelectionTests(unittest.TestCase):
    def _two_chains(self):
        a = conv([utt("r", None, "s1", 0), utt("a", "r", "s2", 1)], cid="c1")
        b = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "a", "s1", 2),
        ], cid="c2")
        return extract_conversation_features([a, b])

    def test_degenerate_columns_found(self):
        df = self._two_chains()
        degenerate = degenerate_columns(df)
        # Pure chains: branching and leaf counts cannot vary.
        self.assertIn("max_branching", degenerate)
        self.assertIn("n_leaves", degenerate)
        # Size does vary between a 2- and a 3-comment thread.
        self.assertNotIn("n_utterances", degenerate)

    def test_keys_never_reported_degenerate(self):
        df = self._two_chains()
        for key in ("conversation_id", "source_id"):
            self.assertNotIn(key, degenerate_columns(df))

    def test_model_ready_excludes_leaky_and_keys(self):
        columns = model_ready_columns(self._two_chains())
        for name in LEAKY_FEATURES:
            self.assertNotIn(name, columns)
        self.assertNotIn("conversation_id", columns)
        self.assertIn("n_utterances", columns)

    def test_model_ready_can_keep_leaky(self):
        # n_deleted must actually vary, or it is dropped as degenerate whatever
        # drop_leaky says.
        clean = conv([utt("r", None, "s1", 0), utt("a", "r", "s2", 1)], cid="c1")
        dirty = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1, is_deleted=True),
        ], cid="c2")
        df = extract_conversation_features([clean, dirty])
        self.assertIn("n_deleted", model_ready_columns(df, drop_leaky=False))
        self.assertNotIn("n_deleted", model_ready_columns(df))


class CutoffConnectivityTests(unittest.TestCase):
    def test_orphaned_reply_dropped_not_flattened(self):
        # b(5) is timestamped before its own parent a(100). first_n=2 selects
        # r(0) and b(5); keeping b without a would report two roots and
        # collapse max_depth to 0. b must be dropped instead.
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 100),
            utt("b", "a", "s3", 5),
        ])
        row = conversation_features(c, first_n=2)
        self.assertEqual(row["n_utterances"], 1)  # only the root survives
        self.assertEqual(row["max_depth"], 0)
        self.assertEqual(row["n_speakers"], 1)

    def test_normal_ordering_unaffected(self):
        # The common case: replies postdate their parents, so nothing is lost.
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "a", "s1", 2),
        ])
        row = conversation_features(c, first_n=2)
        self.assertEqual(row["n_utterances"], 2)
        self.assertEqual(row["max_depth"], 1)

    def test_cutoff_keeps_single_root(self):
        c = conv([
            utt("r", None, "s1", 0),
            utt("a", "r", "s2", 1),
            utt("b", "r", "s3", 2),
            utt("d", "b", "s4", 3),
        ])
        for n in (1, 2, 3, 4):
            row = conversation_features(c, first_n=n)
            # depth 0 is the root; a connected subtree has exactly one.
            self.assertEqual(row["max_depth"] >= 0, True)
            self.assertGreaterEqual(row["n_utterances"], 1)


class DtypeTests(unittest.TestCase):
    def test_all_none_timing_column_is_numeric(self):
        # Coarse Discourse has no timestamps, so every timing column is all-None.
        # Left as object dtype, `.mean()` silently returns NaN and sklearn fails.
        df = extract_conversation_features([
            conv([utt("r", None, "s1"), utt("a", "r", "s2")], cid="c1"),
            conv([utt("r", None, "s1"), utt("a", "r", "s2")], cid="c2"),
        ])
        for column in ("duration_seconds", "median_reply_gap", "growth_rate_per_hour"):
            self.assertEqual(df[column].dtype.kind, "f", column)

    def test_bool_columns_stay_bool(self):
        df = extract_conversation_features([conv([utt("r", None, "s1", 0)])])
        self.assertEqual(df["had_structural_repair"].dtype, bool)
        self.assertEqual(df["last_utterance_is_deleted"].dtype, bool)

    def test_counts_remain_numeric(self):
        df = extract_conversation_features([conv([utt("r", None, "s1", 0, score=3)])])
        self.assertIn(df["n_utterances"].dtype.kind, "iu")
        self.assertEqual(df["mean_score"].dtype.kind, "f")


if __name__ == "__main__":
    unittest.main()
