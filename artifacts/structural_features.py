"""Conversation-level structural features (Task #4b).

One row per Conversation, describing the *shape* of the thread rather than the
content of its text. Text-level features live in `text_features.py` (Task #4a);
the two tables join on `conversation_id`.

Synthetic placeholder utterances are excluded. They were inserted during
standardization to repair missing-parent references and are not real messages, so
counting them would inflate size and depth with invented structure. When a real
reply's parent was a placeholder, it is re-attached to its nearest real ancestor
so the tree stays connected.

Timing features require `created_at`. CGA-CMV has complete timestamps; Coarse
Discourse has none, so those columns come back as None there.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable

import pandas as pd

from schema import Conversation, Utterance

# Features that can leak the CGA-CMV outcome label. `has_removed_comment` is
# defined as "a moderator removed a comment", so any deletion signal computed
# over the whole thread partly restates the answer. Keep them for exploration,
# drop them before training, or compute them under a `first_n` cutoff.
LEAKY_FEATURES = (
    "n_deleted",
    "deleted_ratio",
    "last_utterance_is_deleted",
)

KEY_COLUMNS = ("conversation_id", "source_id")
BOOL_COLUMNS = ("had_structural_repair", "last_utterance_is_deleted")


def _is_synthetic(utterance: Utterance) -> bool:
    return bool(utterance.metadata.get("is_synthetic", False))


def _real_tree(conversation: Conversation) -> tuple[list[Utterance], dict[str, list[Utterance]]]:
    """Return real utterances plus a child map with placeholders bypassed.

    A real reply whose parent is synthetic is re-attached to its nearest real
    ancestor. Utterances with no real ancestor are treated as roots.
    """
    by_id = {u.id: u for u in conversation.utterances}
    real = [u for u in conversation.utterances if not _is_synthetic(u)]
    real_ids = {u.id for u in real}

    children: dict[str, list[Utterance]] = {}
    for utterance in real:
        parent_id = utterance.parent_id
        # Walk up through any chain of placeholders to the first real ancestor.
        while parent_id is not None and parent_id not in real_ids:
            parent = by_id.get(parent_id)
            parent_id = parent.parent_id if parent is not None else None
        if parent_id is not None:
            children.setdefault(parent_id, []).append(utterance)
    return real, children


def _depths(real: list[Utterance], children: dict[str, list[Utterance]]) -> dict[str, int]:
    """Recompute depth over the real-only tree. Roots are depth 0."""
    child_ids = {u.id for kids in children.values() for u in kids}
    roots = [u for u in real if u.id not in child_ids]

    depths: dict[str, int] = {}
    stack = [(u, 0) for u in roots]
    while stack:
        utterance, depth = stack.pop()
        if utterance.id in depths:
            continue  # defensive: ignore cycles rather than hanging
        depths[utterance.id] = depth
        for child in children.get(utterance.id, []):
            stack.append((child, depth + 1))
    return depths


def _ordered(real: list[Utterance]) -> list[Utterance]:
    """Real utterances in time order, falling back to list order when undated."""
    if all(u.created_at is not None for u in real):
        return sorted(real, key=lambda u: u.created_at)
    return list(real)


def _shape_features(
    real: list[Utterance],
    children: dict[str, list[Utterance]],
    depths: dict[str, int],
) -> dict[str, Any]:
    n = len(real)
    depth_values = list(depths.values()) or [0]
    reply_counts = [len(children.get(u.id, [])) for u in real]
    per_depth = Counter(depths.values())

    return {
        "n_utterances": n,
        "max_depth": max(depth_values),
        "mean_depth": statistics.fmean(depth_values),
        "max_branching": max(reply_counts) if reply_counts else 0,
        "n_leaves": sum(1 for count in reply_counts if count == 0),
        "max_width": max(per_depth.values()) if per_depth else 0,
        # Near 1.0 means a single narrow chain (a two-person duel); near 0 means
        # a wide, shallow discussion.
        "depth_ratio": (max(depth_values) / (n - 1)) if n > 1 else 0.0,
    }


_TIMING_KEYS = (
    "duration_seconds",
    "median_reply_gap",
    "mean_reply_gap",
    "min_reply_gap",
    "max_reply_gap",
    "time_to_first_reply",
    "gap_acceleration",
    "growth_rate_per_hour",
)


def _timing_features(
    ordered: list[Utterance],
    children: dict[str, list[Utterance]],
    root_id: str | None,
) -> dict[str, Any]:
    """Reply-speed features. All None on an undated corpus like Coarse Discourse."""
    empty = dict.fromkeys(_TIMING_KEYS)

    dated = [u for u in ordered if u.created_at is not None]
    if len(dated) < 2:
        return empty

    duration = (dated[-1].created_at - dated[0].created_at).total_seconds()
    empty["duration_seconds"] = duration
    # Comments per hour. Measures how fast the thread accumulated, which the
    # issue calls conversation growth rate.
    if duration > 0:
        empty["growth_rate_per_hour"] = len(dated) / (duration / 3600)

    # Gap between each reply and its own parent, not the previous message
    # overall — sibling branches run concurrently.
    parent_time = {u.id: u.created_at for u in dated}
    gaps: list[float] = []
    first_reply_gap = None
    for parent_id, kids in children.items():
        if parent_id not in parent_time:
            continue
        for child in kids:
            if child.created_at is None:
                continue
            gap = (child.created_at - parent_time[parent_id]).total_seconds()
            gaps.append(gap)
            # How long the root sat before anyone engaged.
            if parent_id == root_id and (first_reply_gap is None or gap < first_reply_gap):
                first_reply_gap = gap

    if not gaps:
        return empty

    empty["median_reply_gap"] = statistics.median(gaps)
    empty["mean_reply_gap"] = statistics.fmean(gaps)
    empty["min_reply_gap"] = min(gaps)
    empty["max_reply_gap"] = max(gaps)
    empty["time_to_first_reply"] = first_reply_gap

    # >1 means the exchange slowed down; <1 means it sped up toward the end.
    # Rapid-fire late replies are a plausible heat signal.
    half = len(gaps) // 2
    if half:
        first = statistics.median(gaps[:half])
        second = statistics.median(gaps[half:])
        if first > 0:
            empty["gap_acceleration"] = second / first

    return empty


def _speaker_features(ordered: list[Utterance], children: dict[str, list[Utterance]]) -> dict[str, Any]:
    n = len(ordered)
    counts = Counter(u.speaker_id for u in ordered)
    top_two = sum(count for _, count in counts.most_common(2))

    parent_speaker = {u.id: u.speaker_id for u in ordered}
    self_replies = sum(
        1
        for parent_id, kids in children.items()
        for child in kids
        if parent_speaker.get(parent_id) == child.speaker_id
    )

    return {
        "n_speakers": len(counts),
        "top_speaker_share": (counts.most_common(1)[0][1] / n) if n else 0.0,
        # High share with exactly two speakers is the classic derailment shape.
        "top_two_speaker_share": (top_two / n) if n else 0.0,
        "self_reply_count": self_replies,
        "utterances_per_speaker": (n / len(counts)) if counts else 0.0,
    }


def _signal_features(ordered: list[Utterance], root_id: str | None) -> dict[str, Any]:
    n = len(ordered)
    scores = [u.score for u in ordered if u.score is not None]
    n_deleted = sum(1 for u in ordered if u.is_deleted)

    row: dict[str, Any] = {
        "n_deleted": n_deleted,  # LEAKY
        "deleted_ratio": (n_deleted / n) if n else 0.0,  # LEAKY
        "last_utterance_is_deleted": bool(ordered[-1].is_deleted) if ordered else False,  # LEAKY
        "min_score": min(scores) if scores else None,
        "mean_score": statistics.fmean(scores) if scores else None,
        "n_negative_score": sum(1 for s in scores if s < 0),
        "n_missing_score": n - len(scores),
    }

    # Vote *trajectory*, not just level: the issue asks for patterns in
    # upvotes/downvotes, and n_negative_score is the strongest single signal so
    # far, so it is worth knowing where in the thread the trouble sits.
    row.update(dict.fromkeys(("root_score", "last_score", "score_trend", "min_score_position")))
    if not scores:
        return row

    root = next((u for u in ordered if u.id == root_id), None)
    row["root_score"] = root.score if root is not None else None
    row["last_score"] = ordered[-1].score

    dated_scores = [u.score for u in ordered if u.score is not None]
    half = len(dated_scores) // 2
    if half:
        # Negative means reception got worse as the thread went on.
        row["score_trend"] = statistics.fmean(dated_scores[half:]) - statistics.fmean(dated_scores[:half])

    # Normalized position of the worst-received comment: 0.0 at the start of the
    # thread, 1.0 at the end. Late lows suggest deterioration rather than a
    # thread that began badly.
    scored = [(i, u.score) for i, u in enumerate(ordered) if u.score is not None]
    worst_index = min(scored, key=lambda pair: pair[1])[0]
    row["min_score_position"] = (worst_index / (n - 1)) if n > 1 else 0.0
    return row


def conversation_features(
    conversation: Conversation,
    first_n: int | None = None,
) -> dict[str, Any]:
    """Compute structural features for one conversation.

    Args:
        conversation: the standardized Conversation.
        first_n: when set, use only the earliest `first_n` real utterances. Use
            this for derailment prediction so the model cannot see the outcome;
            see the leakage guidance in `data/STANDARDIZATION.md`.
    """
    real, children = _real_tree(conversation)
    n_synthetic = sum(1 for u in conversation.utterances if _is_synthetic(u))

    if first_n is not None:
        selected = {u.id for u in _ordered(real)[:first_n]}
        # Keep only the part still connected to the root. Selecting the earliest
        # N by timestamp can otherwise orphan a reply whose parent fell outside
        # the cutoff, which would silently flatten depth to 0. Reddit replies
        # normally postdate their parent so this rarely bites, but a corpus with
        # out-of-order timestamps would hit it.
        full_depths = _depths(real, children)
        root = next((u.id for u in _ordered(real) if full_depths.get(u.id) == 0), None)
        kept: set[str] = set()
        if root in selected:
            queue = [root]
            while queue:
                current = queue.pop()
                kept.add(current)
                for child in children.get(current, []):
                    if child.id in selected and child.id not in kept:
                        queue.append(child.id)

        real = [u for u in real if u.id in kept]
        children = {
            parent_id: [c for c in kids if c.id in kept]
            for parent_id, kids in children.items()
            if parent_id in kept
        }

    depths = _depths(real, children)
    ordered = _ordered(real)
    # Root of the retained set, which a `first_n` cutoff leaves unchanged.
    root_id = next((u.id for u in ordered if depths.get(u.id) == 0), None)

    row: dict[str, Any] = {
        "conversation_id": conversation.id,
        "source_id": conversation.source_id,
    }
    row.update(_shape_features(real, children, depths))
    row.update(_timing_features(ordered, children, root_id))
    row.update(_speaker_features(ordered, children))
    row.update(_signal_features(ordered, root_id))
    row["n_synthetic_removed"] = n_synthetic
    row["had_structural_repair"] = n_synthetic > 0
    return row


def extract_conversation_features(
    conversations: Iterable[Conversation],
    first_n: int | None = None,
) -> pd.DataFrame:
    """Build the conversation-level feature table, one row per conversation.

    Named for its grain so it cannot be confused with `text_features.py`, which
    returns one row per comment. Task #6 joins the two tables on
    `conversation_id`.
    """
    rows = [conversation_features(c, first_n=first_n) for c in conversations]
    features = pd.DataFrame(rows)

    # Force numeric dtypes. A column that is entirely None — every timing
    # feature on the undated Coarse Discourse corpus — otherwise lands as
    # object, where `.mean()` quietly returns NaN instead of failing and
    # scikit-learn rejects the column outright.
    for column in features.columns:
        if column in KEY_COLUMNS or column in BOOL_COLUMNS:
            continue
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def degenerate_columns(features: pd.DataFrame) -> list[str]:
    """Columns with no usable variation, so Task #6 can drop them.

    Which columns are degenerate is corpus-dependent, not fixed: every CGA-CMV
    thread is a single linear chain, so all branching and width features are
    constant there, while Coarse Discourse has real trees and they vary.
    """
    return [
        column
        for column in features.columns
        if column not in KEY_COLUMNS and features[column].nunique(dropna=True) <= 1
    ]


def model_ready_columns(features: pd.DataFrame, drop_leaky: bool = True) -> list[str]:
    """Feature columns worth training on: variable, and optionally non-leaky."""
    drop = set(degenerate_columns(features)) | set(KEY_COLUMNS)
    if drop_leaky:
        drop |= set(LEAKY_FEATURES)
    return [column for column in features.columns if column not in drop]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    REPO_ROOT = next(
        path
        for path in [Path.cwd(), *Path.cwd().parents]
        if (path / "pyproject.toml").exists()
    )
    sys.path.insert(0, str(REPO_ROOT / "artifacts"))
    from standardize_convokit import iter_standardized

    source = REPO_ROOT / "data" / "local" / "standardized-v1" / "conversations-gone-awry-cmv-corpus.jsonl"
    out_dir = REPO_ROOT / "data" / "local"

    full = extract_conversation_features(iter_standardized(source))
    full.to_csv(out_dir / "structural_features_cmv.csv", index=False)

    early = extract_conversation_features(iter_standardized(source), first_n=4)
    early.to_csv(out_dir / "structural_features_cmv_first4.csv", index=False)

    print(f"full thread : {full.shape[0]} rows x {full.shape[1]} cols")
    print(f"first 4 only: {early.shape[0]} rows x {early.shape[1]} cols")
    print(f"\nleaky columns (drop before training): {', '.join(LEAKY_FEATURES)}")
    print(f"\nwrote {out_dir / 'structural_features_cmv.csv'}")
    print(f"wrote {out_dir / 'structural_features_cmv_first4.csv'}")
