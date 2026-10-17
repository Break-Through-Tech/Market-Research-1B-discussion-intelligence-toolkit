"""Baseline conversation-quality model (Task #6).

Scope: "quality" here means **civility**, the first dimension of the team's
quality rubric, measured at two levels on CGA-CMV:

* **Reply level.** Is this reply the kind CMV moderators remove under Rule 2
  ("Don't be rude or hostile")? Labels: `civil` / `uncivil`.
* **Conversation level.** Judging from its opening, does the conversation stay
  `civil` or end with such a removal (`derailed`)? Reply-level scores of the
  opening comments are one of its inputs.

At both levels the predicted probability of `civil` is the `quality_score` in
[0.0, 1.0]. It does not measure persuasiveness, relevance or reasoning: CGA-CMV
has no labels for those.

Leakage rules, from `data/STANDARDIZATION.md`:

* In a derailed thread the final utterance is the removed comment, so the model
  only ever sees the context *before* the final utterance.
* The corpus's own train/val/test split is used. It keeps each derailed thread
  and its matched civil partner (same `pair_id` pairing) on the same side.
* Whole-thread deletion counts are dropped (`structural_features.LEAKY_FEATURES`).
* Vote scores are kept separate (`VOTE_FEATURES`): they were retrieved after the
  thread ended, so they can reflect the derailment itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from schema import Conversation, Utterance
from standardize_convokit import iter_standardized
from structural_features import conversation_features, model_ready_columns
from text_features import caps_ratio, exclaim_count, has_question, has_quote, has_url, word_count

CIVIL = "civil"
DERAILED = "derailed"  # conversation-level negative label
UNCIVIL = "uncivil"  # reply-level negative label

# Task #4a per-comment features, used by the simplest reply-level model.
TEXT_FEATURE_COLUMNS = (
    "word_count",
    "char_count",
    "has_question",
    "exclaim_count",
    "caps_ratio",
    "has_quote",
    "has_url",
)

# Per-conversation summaries of the reply model's scores on the opening comments.
REPLY_SCORE_COLUMNS = ("reply_min_civil", "reply_mean_civil", "reply_last_civil")

# Reddit scores are snapshots taken when the data was scraped, after the thread
# ended, so a comment's score can include downvotes earned by the fight that
# followed it. Reported separately rather than trusted by default.
VOTE_FEATURES = (
    "min_score",
    "mean_score",
    "n_negative_score",
    "n_missing_score",
    "root_score",
    "last_score",
    "score_trend",
    "min_score_position",
)


# ---------------------------------------------------------------------------
# Building examples
# ---------------------------------------------------------------------------


def _real_ordered(conversation: Conversation) -> list[Utterance]:
    real = [u for u in conversation.utterances if not u.metadata.get("is_synthetic")]
    if all(u.created_at is not None for u in real):
        return sorted(real, key=lambda u: u.created_at)
    return real


def context_utterances(conversation: Conversation) -> list[Utterance]:
    """Real utterances in time order, minus the final one (the outcome)."""
    return _real_ordered(conversation)[:-1]


def conversation_example(conversation: Conversation) -> dict[str, Any]:
    """One model row: label, split, context text, and pre-outcome structural features."""
    source = conversation.metadata.get("source_metadata", {})
    context = context_utterances(conversation)
    row: dict[str, Any] = {
        "conversation_id": conversation.id,
        "pair_id": source.get("pair_id"),
        "split": source.get("split"),
        "label": DERAILED if source.get("has_removed_comment") else CIVIL,
        "n_context": len(context),
        "context_text": "\n\n".join(u.text_clean for u in context),
    }
    features = conversation_features(conversation, first_n=len(context))
    row.update({k: v for k, v in features.items() if k not in ("conversation_id", "source_id")})
    return row


def load_examples(path: str | Path) -> pd.DataFrame:
    """Load standardized CGA-CMV into one row per conversation."""
    examples = pd.DataFrame(conversation_example(c) for c in iter_standardized(path))
    # Same coercion as structural_features.extract_conversation_features: all-None
    # columns would otherwise be object dtype, which scikit-learn rejects.
    skip = {"conversation_id", "pair_id", "split", "label", "context_text"}
    for column in examples.columns:
        if column not in skip:
            examples[column] = pd.to_numeric(examples[column], errors="coerce")
    return examples


def structural_columns(examples: pd.DataFrame, include_votes: bool) -> list[str]:
    """Usable structural feature columns, with leaky ones always dropped."""
    meta = ["pair_id", "split", "label", "n_context", "context_text", *REPLY_SCORE_COLUMNS]
    columns = model_ready_columns(examples.drop(columns=meta, errors="ignore"), drop_leaky=True)
    if not include_votes:
        columns = [c for c in columns if c not in VOTE_FEATURES]
    return columns


def _text_feature_row(text: str) -> dict[str, float]:
    return {
        "word_count": word_count(text),
        "char_count": len(text),
        "has_question": float(has_question(text)),
        "exclaim_count": exclaim_count(text),
        "caps_ratio": caps_ratio(text),
        "has_quote": float(has_quote(text)),
        "has_url": float(has_url(text)),
    }


def load_replies(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reply-level data: (labeled final replies, unlabeled opening replies).

    Only final replies have labels. In a derailed thread the final reply is the
    one moderators removed under Rule 2, so it is `uncivil`; the final reply of
    its matched civil thread is `civil`. Opening replies (everything before the
    final one) are unlabeled; the reply model scores them for the conversation
    model. Replies that are only a `[deleted]` marker carry no text and are
    dropped from both.
    """
    finals: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    for conversation in iter_standardized(path):
        source = conversation.metadata.get("source_metadata", {})
        ordered = [u for u in _real_ordered(conversation) if not u.is_deleted and u.text_clean.strip()]
        final = _real_ordered(conversation)[-1]
        if ordered and ordered[-1] is final:
            finals.append({
                "conversation_id": conversation.id,
                "pair_id": source.get("pair_id"),
                "split": source.get("split"),
                "label": UNCIVIL if source.get("has_removed_comment") else CIVIL,
                "text": final.text_clean,
                **_text_feature_row(final.text_clean),
            })
        openings.extend(
            {"conversation_id": conversation.id, "text": u.text_clean}
            for u in ordered
            if u is not final
        )
    return pd.DataFrame(finals), pd.DataFrame(openings)


def _pair_key(conversation_id: str, pair_id: str) -> str:
    return "|".join(sorted([str(conversation_id).rsplit(":", 1)[-1], str(pair_id)]))


def add_reply_scores(
    examples: pd.DataFrame,
    finals: pd.DataFrame,
    openings: pd.DataFrame,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Add REPLY_SCORE_COLUMNS: the reply model's civility scores, summarized per conversation.

    Train conversations are scored *out-of-fold*: the reply model that scores
    a train conversation never saw that conversation's (or its pair partner's)
    final reply. Otherwise the conversation model would train on scores that
    already know the answer. Val/test use one reply model fit on all of train.
    """
    train_finals = finals[finals["split"] == "train"]
    is_train = examples["split"] == "train"
    train_ids = examples.loc[is_train, "conversation_id"].to_numpy()
    scores = pd.Series(np.nan, index=openings.index)

    def score(fit_ids: set[str], score_ids: set[str]) -> None:
        fit = train_finals[train_finals["conversation_id"].isin(fit_ids)]
        model = text_model().fit(fit["text"], fit["label"])
        rows = openings["conversation_id"].isin(score_ids)
        if rows.any():
            scores.loc[rows] = model.quality_scores(openings.loc[rows, "text"])

    groups = [_pair_key(c, p) for c, p in zip(train_ids, examples.loc[is_train, "pair_id"])]
    for fit_index, score_index in GroupKFold(n_splits=n_folds).split(train_ids, groups=groups):
        score(set(train_ids[fit_index]), set(train_ids[score_index]))
    score(set(train_ids), set(examples.loc[~is_train, "conversation_id"]))

    summary = openings.assign(civil=scores).groupby("conversation_id", sort=False)["civil"].agg(
        reply_min_civil="min",  # the single worst opening reply
        reply_mean_civil="mean",
        reply_last_civil="last",  # the reply just before the outcome
    )
    return examples.merge(summary, left_on="conversation_id", right_index=True, how="left")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class QualityModel:
    """A scikit-learn pipeline that predicts `civil` vs `derailed`/`uncivil` labels.

    `predict` matches evaluation.py's `TextClassifier` protocol. The text model
    takes a list of strings; the feature model takes a DataFrame of feature rows
    instead, since thread shape cannot be recovered from a string. Both work with
    `evaluate_classifier`, which only calls `.predict()`.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def fit(self, X: Any, labels: Sequence[str]) -> "QualityModel":
        self.pipeline.fit(X, list(labels))
        return self

    def predict(self, X: Any) -> list[str]:
        return [str(label) for label in self.pipeline.predict(X)]

    def quality_scores(self, X: Any) -> list[float]:
        """Probability of `civil` for each input: the `quality_score`."""
        civil_index = list(self.pipeline.classes_).index(CIVIL)
        return self.pipeline.predict_proba(X)[:, civil_index].tolist()


def majority_baseline() -> QualityModel:
    """Always predicts the most common training label. The floor to beat.

    Its quality score is the training base rate of `civil` for every thread.
    """
    return QualityModel(Pipeline([("majority", DummyClassifier(strategy="prior"))]))


def feature_model(columns: Sequence[str]) -> QualityModel:
    """Logistic regression on numeric feature columns (structural, reply scores, Task #4a)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return QualityModel(Pipeline([
        ("features", ColumnTransformer([("numeric", numeric, list(columns))])),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]))


def text_model() -> QualityModel:
    """TF-IDF + logistic regression on text (a reply, or a conversation's opening)."""
    return QualityModel(Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def paired_accuracy(
    quality_scores: Sequence[float],
    labels: Sequence[str],
    pair_ids: Sequence[str],
    conversation_ids: Sequence[str],
) -> float:
    """Share of matched pairs where the civil item outscores its partner.

    CGA-CMV pairs every derailed thread with a civil thread from the same post,
    so this asks the fairest version of the question: same topic, which one goes
    wrong? Works for conversations and for their final replies. 0.5 is chance.
    Ties count as half. `pair_id` holds the partner's source ID, so pairs are
    matched on the ID's last segment.
    """
    by_source = {
        str(cid).rsplit(":", 1)[-1]: (score, label)
        for cid, score, label in zip(conversation_ids, quality_scores, labels)
    }
    wins: list[float] = []
    for cid, pair_id, label in zip(conversation_ids, pair_ids, labels):
        if label != CIVIL or pair_id not in by_source:
            continue
        civil_score = by_source[str(cid).rsplit(":", 1)[-1]][0]
        derailed_score, partner_label = by_source[pair_id]
        if partner_label == CIVIL:
            continue
        wins.append(1.0 if civil_score > derailed_score else 0.5 if civil_score == derailed_score else 0.0)
    return float(np.mean(wins)) if wins else float("nan")
