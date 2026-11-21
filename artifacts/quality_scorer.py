"""Lightweight comment-quality scorer (November Task #1 / issue #29).

The LLM labeler (`quality_labeler.py`) is accurate but slow and costs usage, so
it labels a sample. This module trains a cheap model on those LLM labels so the
pipeline can score every comment in seconds.

For each rubric dimension (civility, relevance, reasoning, constructiveness) a
ridge regression predicts a continuous score on the rubric's 0-2 scale from:

* TF-IDF word 1-2 grams of the reply, and
* a few hand-made features: length, question marks, second-person words, and
  how similar the reply is to its parent and to the thread start (TF-IDF
  cosine), which mainly helps relevance.

Reasoning may be "N/A" (the reply makes no claim). The scorer handles it in two
parts: a logistic regression gives `reasoning_na_prob`, and the ridge for the
reasoning score is trained only on replies that had a numeric label.

Pipeline use::

    scorer = QualityScorer.load("data/local/quality_scorer.joblib")
    scores = scorer.score(df)          # df has a `reply` column
    # or, for plain strings:
    scores = score_comments(["text one", "text two"], scorer=scorer)

`score` returns one row per input with `<dimension>_score` (float, clipped to
0-2), `reasoning_na_prob`, and `overall_score` (mean of the dimensions in
`OVERALL_DIMENSIONS`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import paired_cosine_distances

from quality_labeler import DIMENSIONS, NA, SAME_AS_START

MIN_SCORE, MAX_SCORE = 0.0, 2.0
# Relevance is left out of the overall score: almost every reply is on topic
# (94% of LLM labels and 57 of 60 human labels are 2), so it adds little signal.
OVERALL_DIMENSIONS = ("civility", "reasoning", "constructiveness")

_SECOND_PERSON = re.compile(r"\b(you|your|you're|youre|yourself|u)\b", re.IGNORECASE)
_WORD = re.compile(r"\w+")


def _text(values: Iterable[Any]) -> list[str]:
    return ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values]


def _numeric_labels(series: pd.Series) -> pd.Series:
    """Labels as floats; "N/A" and blanks become NaN."""
    return pd.to_numeric(series.astype(str).str.strip().replace({NA: None, "": None}), errors="coerce")


class QualityScorer:
    """TF-IDF + ridge scorer for the four rubric dimensions."""

    def __init__(self, alpha: float = 3.0, max_features: int = 30_000, min_df: int = 2):
        self.alpha = alpha
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer: TfidfVectorizer | None = None
        self.regressors: dict[str, Ridge] = {}
        self.na_classifier: LogisticRegression | None = None
        self.na_constant: float | None = None  # used when training data has one class only
        self.extra_mean: np.ndarray | None = None
        self.extra_std: np.ndarray | None = None

    # -- features -----------------------------------------------------------

    def _extra(self, frame: pd.DataFrame) -> np.ndarray:
        reply = _text(frame["reply"])
        parent = _text(frame["parent"]) if "parent" in frame else [""] * len(reply)
        start = _text(frame["thread_start"]) if "thread_start" in frame else [""] * len(reply)
        parent = [s if p == SAME_AS_START else p for p, s in zip(parent, start)]

        def similarity(other: list[str]) -> np.ndarray:
            a, b = self.vectorizer.transform(reply), self.vectorizer.transform(other)
            sim = 1.0 - paired_cosine_distances(a, b)
            has_other = np.array([bool(o.strip()) for o in other])
            return np.where(has_other, sim, 0.0)

        words = np.array([len(_WORD.findall(r)) for r in reply], dtype=float)
        return np.column_stack([
            np.log1p(words),
            [r.count("?") for r in reply],
            [len(_SECOND_PERSON.findall(r)) / max(n, 1) for r, n in zip(reply, words)],
            similarity(parent),
            similarity(start),
        ])

    def _features(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        extra = (self._extra(frame) - self.extra_mean) / self.extra_std
        return sparse.hstack([self.vectorizer.transform(_text(frame["reply"])), sparse.csr_matrix(extra)]).tocsr()

    # -- fit / predict ------------------------------------------------------

    def fit(self, frame: pd.DataFrame, labels: pd.DataFrame | None = None) -> "QualityScorer":
        """Train on replies and their labels.

        `frame` needs `reply` (and ideally `parent`, `thread_start`) plus the four
        dimension columns, or pass the labels separately as `labels` aligned to
        `frame` by position.
        """
        if labels is not None:
            frame = pd.concat([frame.reset_index(drop=True), labels[list(DIMENSIONS)].reset_index(drop=True)], axis=1)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_features,
            sublinear_tf=True, strip_accents="unicode",
        )
        self.vectorizer.fit(_text(frame["reply"]))
        extra = self._extra(frame)
        self.extra_mean = extra.mean(axis=0)
        self.extra_std = np.where(extra.std(axis=0) > 0, extra.std(axis=0), 1.0)
        X = self._features(frame)

        self.regressors = {}
        for dim in DIMENSIONS:
            y = _numeric_labels(frame[dim])
            keep = y.notna().to_numpy()
            if not keep.any():
                raise ValueError(f"no numeric labels for {dim}")
            self.regressors[dim] = Ridge(alpha=self.alpha).fit(X[keep], y[keep].to_numpy())

        is_na = (frame["reasoning"].astype(str).str.strip() == NA).to_numpy()
        if is_na.all() or not is_na.any():
            self.na_classifier, self.na_constant = None, float(is_na.mean())
        else:
            self.na_classifier = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, is_na)
            self.na_constant = None
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scores for each row of `frame` (needs a `reply` column)."""
        if self.vectorizer is None:
            raise RuntimeError("scorer is not trained; call fit() or QualityScorer.load()")
        X = self._features(frame)
        out = pd.DataFrame(index=frame.index)
        for dim in DIMENSIONS:
            out[f"{dim}_score"] = np.clip(self.regressors[dim].predict(X), MIN_SCORE, MAX_SCORE)
        if self.na_classifier is not None:
            out["reasoning_na_prob"] = self.na_classifier.predict_proba(X)[:, 1]
        else:
            out["reasoning_na_prob"] = self.na_constant
        out["overall_score"] = out[[f"{d}_score" for d in OVERALL_DIMENSIONS]].mean(axis=1)
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QualityScorer":
        scorer = joblib.load(path)
        if not isinstance(scorer, cls):
            raise TypeError(f"{path} does not hold a QualityScorer")
        return scorer


def score_comments(
    replies: Sequence[str],
    scorer: QualityScorer,
    parents: Sequence[str] | None = None,
    thread_starts: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score plain strings. `parents` / `thread_starts` are optional context."""
    frame = pd.DataFrame({"reply": list(replies)})
    if parents is not None:
        frame["parent"] = list(parents)
    if thread_starts is not None:
        frame["thread_start"] = list(thread_starts)
    return scorer.score(frame)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_scores(reference: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """MAE and Spearman per dimension, comparing scores with reference labels.

    `reference` has `utterance_id` and the dimension columns (0/1/2/"N/A").
    `predicted` has `utterance_id` and either `<dim>_score` floats (scorer) or
    the dimension columns themselves (e.g. LLM labels). Rows whose reference is
    N/A are left out of that dimension's MAE/Spearman; for reasoning,
    `na_accuracy` reports how often N/A was called correctly (threshold 0.5).
    """
    predicted = predicted.copy()
    for dim in DIMENSIONS:
        if f"{dim}_score" not in predicted:
            predicted[f"{dim}_score"] = _numeric_labels(predicted[dim])
    if "reasoning_na_prob" not in predicted and "reasoning" in predicted:
        predicted["reasoning_na_prob"] = (predicted["reasoning"].astype(str).str.strip() == NA).astype(float)
    cols = ["utterance_id", *[f"{d}_score" for d in DIMENSIONS], "reasoning_na_prob"]
    merged = reference.merge(predicted[cols], on="utterance_id")

    rows = []
    for dim in DIMENSIONS:
        ref = _numeric_labels(merged[dim])
        pred = merged[f"{dim}_score"].astype(float)
        keep = ref.notna() & pred.notna()
        r, p = ref[keep], pred[keep]
        rho = spearmanr(r, p).statistic if r.nunique() > 1 and p.nunique() > 1 else float("nan")
        row = {
            "dimension": dim,
            "n": int(keep.sum()),
            "mae": float((r - p).abs().mean()) if len(r) else float("nan"),
            "mae_rounded": float((r - p.round()).abs().mean()) if len(r) else float("nan"),
            "spearman": float(rho),
            "mae_predict_mean": float((r - r.mean()).abs().mean()) if len(r) else float("nan"),
        }
        if dim == "reasoning":
            ref_na = merged["reasoning"].astype(str).str.strip() == NA
            pred_na = merged["reasoning_na_prob"].astype(float) >= 0.5
            row["na_accuracy"] = float((ref_na == pred_na).mean()) if len(merged) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def large_errors(
    reference: pd.DataFrame,
    predicted: pd.DataFrame,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """Rows where any dimension's score is at least `threshold` off the reference."""
    cols = ["utterance_id", *[f"{d}_score" for d in DIMENSIONS], "reasoning_na_prob"]
    merged = reference.merge(predicted[cols], on="utterance_id")
    flagged = pd.Series(False, index=merged.index)
    for dim in DIMENSIONS:
        err = (_numeric_labels(merged[dim]) - merged[f"{dim}_score"]).abs()
        merged[f"{dim}_error"] = err
        flagged |= err >= threshold
    return merged[flagged]

