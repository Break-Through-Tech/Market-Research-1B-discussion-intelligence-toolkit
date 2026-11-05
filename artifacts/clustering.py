"""Theme clustering over cached sentence embeddings (Task #7).

Groups comments by what they are *about*. Task #7 of the embedding comparison
found that sentence embeddings mostly encode topic rather than discourse act,
which is why they underperformed TF-IDF there and why they are the right input
here.

Two scopes, deliberately handled by different methods:

  * Corpus scope — ~115k comments. Reduce to a low dimension, then HDBSCAN,
    which picks its own cluster count and may label a comment as noise.
  * Thread scope — tens of comments. Density estimation needs more points than a
    single thread has, so agglomerative clustering with a distance cutoff is
    used instead.

The KMeans baseline from `notebooks/06_embedding_comparison.ipynb` scored ~0.02
silhouette on both corpora with k=10. That is the number to beat, and
`kmeans_baseline` reproduces it so comparisons stay honest.

Silhouette is a weak signal on its own: it rewards tight geometry, not themes a
reader would recognize. Always pair it with `representative_examples` and read
the output.

Dimensionality reduction offers PCA (already available via scikit-learn) and
UMAP (`uv add umap-learn`). PCA is the no-dependency way to check whether
reducing dimensions helps before the team commits to UMAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from embeddings import CachedEmbeddings, cache_dir, load_cache
from evaluation import clustering_metrics

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# HDBSCAN marks points it will not assign to any cluster.
NOISE_LABEL = -1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class ClusterInput:
    """Vectors ready to cluster, plus the ids/text metadata for each row."""

    table: pd.DataFrame
    vectors: np.ndarray

    def __len__(self) -> int:
        return len(self.table)


def load_corpus_vectors(
    corpus: str,
    model_name: str = DEFAULT_MODEL,
    cache_root: Path | None = None,
    drop_unusable: bool = True,
) -> ClusterInput:
    """Load a cached corpus and return the rows worth clustering.

    `drop_unusable` removes empty and deleted comments. They carry no topic and
    collapse into one large meaningless cluster if left in, which also drags the
    silhouette score around. The flags come from the cache, so this needs no
    access to the source corpora.
    """
    directory = cache_dir(model_name, corpus) if cache_root is None else cache_dir(model_name, corpus, cache_root)
    cached = load_cache(directory)
    return _select(cached, drop_unusable=drop_unusable)


def _select(cached: CachedEmbeddings, drop_unusable: bool) -> ClusterInput:
    table = cached.ids.reset_index(drop=True)
    vectors = np.asarray(cached.embeddings)
    if drop_unusable:
        keep = (~table["is_empty"].to_numpy()) & (~table["is_deleted"].to_numpy())
        table = table.loc[keep].reset_index(drop=True)
        vectors = vectors[keep]
    return ClusterInput(table=table, vectors=vectors)


def filter_by_length(data: ClusterInput, text_table: pd.DataFrame, min_words: int = 15) -> ClusterInput:
    """Keep comments with at least `min_words` whitespace-separated tokens.

    Short comments embed by surface form rather than subject: "Thanks!" about a
    graphics card and "Thanks!" about a breakup land in the same region. Left in,
    they produce clusters that are discourse acts (appreciation, humor) rather
    than themes, which is what k=10 KMeans returns on the unfiltered corpus.

    `min_words` matches the `word_count` definition in `text_features.py`, so
    the threshold is comparable to Task #4a's distribution. Pass the table from
    `embeddings.load_text_table(corpus)`.
    """
    words = text_table.set_index("utterance_id")["text"].str.split().str.len()
    counts = data.table["utterance_id"].map(words).fillna(0).to_numpy()
    keep = counts >= min_words
    return ClusterInput(table=data.table.loc[keep].reset_index(drop=True), vectors=data.vectors[keep])


def thread_inputs(data: ClusterInput, min_utterances: int = 5) -> Iterable[tuple[str, ClusterInput]]:
    """Split a corpus into per-conversation inputs, skipping threads too small
    to cluster meaningfully."""
    for conversation_id, group in data.table.groupby("conversation_id", sort=False):
        if len(group) < min_utterances:
            continue
        yield conversation_id, ClusterInput(
            table=group.reset_index(drop=True),
            vectors=data.vectors[group.index.to_numpy()],
        )


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------


#
# 384 dimensions is where density-based clustering breaks down: distances
# concentrate and every point looks roughly equidistant from every other.
# Reducing first is what makes HDBSCAN viable.
#
# Two options, because they differ in cost rather than only in quality:
#
#   PCA  — already available through scikit-learn, fast, deterministic, but
#          limited to linear structure. Use it to test whether reducing
#          dimensions helps at all before committing the team to a dependency.
#   UMAP — follows curved structure and usually separates text embeddings
#          better, at the cost of `uv add umap-learn` and a slower, stochastic
#          fit.
#
# Default `n_components` differs by method on purpose: PCA needs more
# dimensions to retain the same information, UMAP deliberately compresses hard.

DEFAULT_COMPONENTS = {"pca": 50, "umap": 10}


def reduce_pca(vectors: np.ndarray, n_components: int = 50, seed: int = 42) -> np.ndarray:
    """PCA to `n_components` dimensions. No extra dependency, deterministic."""
    return PCA(n_components=n_components, random_state=seed).fit_transform(vectors)


def reduce_umap(
    vectors: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """UMAP to `n_components` dimensions.

    `min_dist=0` is the usual choice when the output feeds a clusterer rather
    than a plot, because it lets points pack tightly.

    UMAP is stochastic; `seed` fixes it but also disables parallelism, so runs
    are reproducible but slower.
    """
    try:
        import umap
    except ImportError as error:  # pragma: no cover - depends on local install
        raise ImportError(
            "UMAP is not installed. Run `uv add umap-learn`, "
            "or use reduce_dimensions(..., method='pca') instead."
        ) from error

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(vectors)


def reduce_dimensions(
    vectors: np.ndarray,
    method: str = "umap",
    n_components: int | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch to `reduce_pca` or `reduce_umap` so the two can be swapped.

    Leaving `n_components` as None picks the default for the chosen method.
    """
    if method not in DEFAULT_COMPONENTS:
        raise ValueError(f"method must be one of {sorted(DEFAULT_COMPONENTS)}, got {method!r}")
    n_components = DEFAULT_COMPONENTS[method] if n_components is None else n_components
    if method == "pca":
        return reduce_pca(vectors, n_components=n_components, seed=seed, **kwargs)
    return reduce_umap(vectors, n_components=n_components, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def kmeans_baseline(vectors: np.ndarray, n_clusters: int = 10, seed: int = 0) -> np.ndarray:
    """The baseline from the embedding comparison notebook (~0.02 silhouette)."""
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(vectors)


def cluster_corpus(
    vectors: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
) -> np.ndarray:
    """HDBSCAN over a whole corpus. Returns labels, with -1 for noise.

    `min_cluster_size` is the main knob: it sets the smallest group that counts
    as a theme, and raising it yields fewer, broader themes. `min_samples`
    controls how conservative the clusterer is; leaving it as None makes it
    equal to `min_cluster_size`.

    `copy=True` because callers sweep several `min_cluster_size` values over one
    reduced array; without it scikit-learn may modify that array in place and
    later settings would see different input. It also silences the FutureWarning
    about this default changing in scikit-learn 1.10.
    """
    return HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True
    ).fit_predict(vectors)


def cluster_thread(vectors: np.ndarray, distance_threshold: float = 0.6) -> np.ndarray:
    """Agglomerative clustering for a single thread.

    HDBSCAN needs more points than one thread provides, so this uses a distance
    cutoff instead of a target cluster count: comments merge while they stay
    closer than `distance_threshold`. Vectors are L2-normalized in the cache, so
    cosine distance is in [0, 2] and the threshold is model-specific — see the
    note in the embedding comparison that bge-small's similarities run higher
    than the other candidates'.
    """
    if len(vectors) < 2:
        return np.zeros(len(vectors), dtype=int)
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(vectors)


# ---------------------------------------------------------------------------
# Scoring and inspection
# ---------------------------------------------------------------------------


def cluster_summary(labels: np.ndarray) -> dict[str, Any]:
    """Shape of a clustering: how many themes, how much was left unassigned."""
    labels = np.asarray(labels)
    assigned = labels[labels != NOISE_LABEL]
    sizes = pd.Series(assigned).value_counts()
    return {
        "n_points": int(len(labels)),
        "n_clusters": int(len(sizes)),
        "noise_rate": float((labels == NOISE_LABEL).mean()),
        "largest_cluster_share": float(sizes.iloc[0] / len(labels)) if len(sizes) else 0.0,
        "median_cluster_size": float(sizes.median()) if len(sizes) else 0.0,
    }


def score_clusters(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette over assigned points only.

    Noise is excluded rather than treated as a cluster: HDBSCAN's noise points
    are by definition scattered, so scoring them as one group would understate a
    clustering that is actually fine. This makes the score comparable to KMeans
    only alongside the noise rate from `cluster_summary` — a clustering that
    assigns 5% of the data can score well and still be useless.
    """
    labels = np.asarray(labels)
    keep = labels != NOISE_LABEL
    if keep.sum() < 2:
        return float("nan")
    return clustering_metrics(np.asarray(vectors)[keep], labels[keep]).silhouette


def representative_examples(
    data: ClusterInput,
    labels: np.ndarray,
    vectors: np.ndarray | None = None,
    n_per_cluster: int = 5,
) -> pd.DataFrame:
    """The `n_per_cluster` points closest to each cluster's centre.

    This is the part that decides whether a clustering is any good. A cluster
    you can give a name to is a success; a strong silhouette over clusters you
    cannot describe is not. Pass `vectors` to rank in the same space used for
    clustering (the reduced one); otherwise the original embeddings are used.
    """
    space = data.vectors if vectors is None else np.asarray(vectors)
    labels = np.asarray(labels)

    rows = []
    for label in sorted(set(labels) - {NOISE_LABEL}):
        members = np.flatnonzero(labels == label)
        centroid = space[members].mean(axis=0)
        distances = np.linalg.norm(space[members] - centroid, axis=1)
        for rank, index in enumerate(members[np.argsort(distances)][:n_per_cluster]):
            row = data.table.iloc[index]
            rows.append({
                "cluster": int(label),
                "cluster_size": int(len(members)),
                "rank": rank,
                "utterance_id": row["utterance_id"],
                "conversation_id": row["conversation_id"],
                "discourse_act": row.get("discourse_act"),
            })
    return pd.DataFrame(rows)


def attach_text(examples: pd.DataFrame, text_table: pd.DataFrame) -> pd.DataFrame:
    """Join comment text onto `representative_examples` output.

    Text is not in the embedding cache, so read it with
    `embeddings.load_text_table(corpus)` when it is time to inspect clusters.
    """
    return examples.merge(
        text_table[["utterance_id", "text"]],
        on="utterance_id",
        how="left",
        validate="many_to_one",
    )
