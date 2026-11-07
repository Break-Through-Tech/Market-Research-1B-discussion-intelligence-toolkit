import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'artifacts'))
from clustering import (
    NOISE_LABEL, ClusterInput, attach_text, cluster_corpus, cluster_summary,
    cluster_thread, filter_by_length, kmeans_baseline, load_corpus_vectors,
    reduce_dimensions, representative_examples, score_clusters, thread_inputs,
)
from embeddings import save_cache


def make_blobs(n_per_blob=40, seed=0):
    """Three well-separated blobs in 3D, L2-normalized like the real cache."""
    rng = np.random.default_rng(seed)
    centres = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    vectors = np.vstack([c + rng.normal(scale=0.1, size=(n_per_blob, 3)) for c in centres])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    truth = np.repeat([0, 1, 2], n_per_blob)
    return vectors.astype(np.float32), truth


def make_table(n, conversation_ids=None, is_empty=None, is_deleted=None):
    return pd.DataFrame({
        "utterance_id": [f"u{i}" for i in range(n)],
        "conversation_id": conversation_ids if conversation_ids is not None else ["c0"] * n,
        "discourse_act": ["answer"] * n,
        "is_deleted": is_deleted if is_deleted is not None else [False] * n,
        "is_empty": is_empty if is_empty is not None else [False] * n,
    })


class TestLoading(unittest.TestCase):
    def test_drops_empty_and_deleted(self):
        vectors, _ = make_blobs(n_per_blob=2)
        table = make_table(
            6,
            is_empty=[False, True, False, False, False, False],
            is_deleted=[False, False, False, True, False, False],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_cache(root / "fake__model" / "coarse-discourse", table, vectors, {})
            data = load_corpus_vectors("coarse-discourse", model_name="fake/model", cache_root=root)

        self.assertEqual(len(data), 4)
        self.assertEqual(len(data.vectors), 4)
        self.assertNotIn("u1", set(data.table["utterance_id"]))
        self.assertNotIn("u3", set(data.table["utterance_id"]))

    def test_keeps_unusable_when_asked(self):
        vectors, _ = make_blobs(n_per_blob=2)
        table = make_table(6, is_empty=[True] * 6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_cache(root / "fake__model" / "cga-cmv", table, vectors, {})
            data = load_corpus_vectors(
                "cga-cmv", model_name="fake/model", cache_root=root, drop_unusable=False
            )
        self.assertEqual(len(data), 6)

    def test_dropped_row_drops_its_own_vector(self):
        # Alignment is the thing most likely to break silently: the row removed
        # from the table must be the row removed from the vectors.
        vectors, _ = make_blobs(n_per_blob=2)
        table = make_table(6, is_empty=[True, False, False, False, False, False])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_cache(root / "fake__model" / "coarse-discourse", table, vectors, {})
            data = load_corpus_vectors("coarse-discourse", model_name="fake/model", cache_root=root)

        np.testing.assert_array_equal(data.vectors, vectors[1:])
        self.assertEqual(list(data.table["utterance_id"]), ["u1", "u2", "u3", "u4", "u5"])


class TestLengthFilter(unittest.TestCase):
    def setUp(self):
        self.vectors, _ = make_blobs(n_per_blob=2)  # 6 rows
        self.data = ClusterInput(table=make_table(6), vectors=self.vectors)
        self.text_table = pd.DataFrame({
            "utterance_id": [f"u{i}" for i in range(6)],
            "text": ["Thanks!", "one two three four five", "", "Thanks. :)",
                     "a b c d e f g", "short"],
        })

    def test_keeps_only_long_enough_comments(self):
        kept = filter_by_length(self.data, self.text_table, min_words=5)
        self.assertEqual(list(kept.table["utterance_id"]), ["u1", "u4"])
        np.testing.assert_array_equal(kept.vectors, self.vectors[[1, 4]])

    def test_missing_text_is_dropped_not_crashed(self):
        # A row absent from the text table has no word count and must not survive.
        kept = filter_by_length(self.data, self.text_table.iloc[:3], min_words=1)
        self.assertEqual(list(kept.table["utterance_id"]), ["u0", "u1"])

    def test_threshold_of_zero_keeps_everything(self):
        kept = filter_by_length(self.data, self.text_table, min_words=0)
        self.assertEqual(len(kept), 6)


class TestThreadInputs(unittest.TestCase):
    def test_skips_short_threads_and_keeps_vectors_aligned(self):
        vectors, _ = make_blobs(n_per_blob=3)  # 9 rows
        table = make_table(9, conversation_ids=["a"] * 6 + ["b"] * 3)
        data = ClusterInput(table=table, vectors=vectors)

        threads = dict(thread_inputs(data, min_utterances=5))

        self.assertEqual(list(threads), ["a"])
        self.assertEqual(len(threads["a"]), 6)
        np.testing.assert_array_equal(threads["a"].vectors, vectors[:6])


class TestReduction(unittest.TestCase):
    def test_pca_returns_requested_components(self):
        vectors, _ = make_blobs(n_per_blob=40)
        reduced = reduce_dimensions(vectors, method="pca", n_components=2)
        self.assertEqual(reduced.shape, (len(vectors), 2))

    def test_pca_is_deterministic(self):
        vectors, _ = make_blobs()
        first = reduce_dimensions(vectors, method="pca", n_components=2)
        second = reduce_dimensions(vectors, method="pca", n_components=2)
        np.testing.assert_allclose(first, second)

    def test_pca_keeps_blobs_separable(self):
        vectors, truth = make_blobs()
        reduced = reduce_dimensions(vectors, method="pca", n_components=2)
        self.assertGreater(score_clusters(reduced, truth), 0.5)

    def test_unknown_method_rejected(self):
        vectors, _ = make_blobs(n_per_blob=5)
        with self.assertRaises(ValueError):
            reduce_dimensions(vectors, method="tsne")


class TestClustering(unittest.TestCase):
    def test_hdbscan_recovers_separated_blobs(self):
        vectors, truth = make_blobs()
        labels = cluster_corpus(vectors, min_cluster_size=10)
        assigned = labels != NOISE_LABEL
        self.assertEqual(len(set(labels[assigned])), 3)
        # Points from the same true blob land together.
        for blob in range(3):
            members = labels[truth == blob]
            self.assertEqual(len(set(members[members != NOISE_LABEL])), 1)

    def test_thread_clustering_handles_tiny_inputs(self):
        self.assertEqual(len(cluster_thread(np.zeros((0, 3)))), 0)
        self.assertEqual(list(cluster_thread(np.ones((1, 3)))), [0])

    def test_thread_clustering_separates_two_topics(self):
        vectors, _ = make_blobs(n_per_blob=4)
        labels = cluster_thread(vectors[:8], distance_threshold=0.5)
        self.assertEqual(len(set(labels)), 2)

    def test_kmeans_baseline_returns_requested_k(self):
        vectors, _ = make_blobs()
        self.assertEqual(len(set(kmeans_baseline(vectors, n_clusters=4))), 4)


class TestScoring(unittest.TestCase):
    def test_summary_reports_noise(self):
        labels = np.array([0, 0, 0, 1, 1, NOISE_LABEL, NOISE_LABEL])
        summary = cluster_summary(labels)
        self.assertEqual(summary["n_clusters"], 2)
        self.assertEqual(summary["n_points"], 7)
        self.assertAlmostEqual(summary["noise_rate"], 2 / 7)
        self.assertAlmostEqual(summary["largest_cluster_share"], 3 / 7)

    def test_summary_handles_all_noise(self):
        summary = cluster_summary(np.array([NOISE_LABEL, NOISE_LABEL]))
        self.assertEqual(summary["n_clusters"], 0)
        self.assertEqual(summary["largest_cluster_share"], 0.0)

    def test_score_excludes_noise(self):
        vectors, truth = make_blobs(n_per_blob=10)
        with_noise = truth.copy()
        with_noise[:3] = NOISE_LABEL
        self.assertGreater(score_clusters(vectors, with_noise), 0.5)

    def test_score_is_nan_without_enough_assigned_points(self):
        vectors, _ = make_blobs(n_per_blob=2)
        labels = np.full(len(vectors), NOISE_LABEL)
        self.assertTrue(np.isnan(score_clusters(vectors, labels)))


class TestExamples(unittest.TestCase):
    def test_returns_closest_members_and_skips_noise(self):
        vectors, truth = make_blobs(n_per_blob=10)
        labels = truth.copy()
        labels[:2] = NOISE_LABEL
        data = ClusterInput(table=make_table(len(vectors)), vectors=vectors)

        examples = representative_examples(data, labels, n_per_cluster=3)

        self.assertEqual(sorted(set(examples["cluster"])), [0, 1, 2])
        self.assertEqual(len(examples), 9)
        self.assertEqual(set(examples["cluster_size"]), {8, 10})

    def test_attach_text_joins_on_utterance_id(self):
        vectors, truth = make_blobs(n_per_blob=5)
        data = ClusterInput(table=make_table(len(vectors)), vectors=vectors)
        examples = representative_examples(data, truth, n_per_cluster=2)
        text_table = pd.DataFrame({
            "utterance_id": [f"u{i}" for i in range(len(vectors))],
            "text": [f"text {i}" for i in range(len(vectors))],
        })

        joined = attach_text(examples, text_table)

        self.assertEqual(len(joined), len(examples))
        self.assertFalse(joined["text"].isna().any())


if __name__ == "__main__":
    unittest.main()
