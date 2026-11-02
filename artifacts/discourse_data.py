"""Baseline-compatible Coarse Discourse examples and auditable split manifests.

Imports follow the repository's artifacts-on-PYTHONPATH convention.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from corpora import ClassificationExample, ClassificationSplits, split_by_group
from standardize_convokit import iter_standardized

CORPUS = "reddit-coarse-discourse-corpus"


def load_discourse_data(path: str | Path, seed: int = 42) -> ClassificationSplits:
    """Match notebook 04: clean text, nine labels, 80/10/10 grouped splits.

    Parent text is metadata only, so context never changes example eligibility.
    """
    examples = []
    seen = set()
    for conversation in iter_standardized(path):
        if not conversation.id.startswith(f"reddit:{CORPUS}:conversation:"):
            raise ValueError("Expected standardized Coarse Discourse conversations")
        by_id = {u.id: u for u in conversation.utterances}
        for message in conversation.utterances:
            if (message.metadata.get("is_synthetic") or message.discourse_act is None
                    or message.discourse_act == "other" or not message.text_clean.strip()):
                continue
            if message.id in seen:
                raise ValueError(f"Duplicate example ID: {message.id}")
            seen.add(message.id)
            parent = by_id.get(message.parent_id)
            parent_text = (parent.text_clean if parent is not None
                           and not parent.metadata.get("is_synthetic") else "")
            examples.append(ClassificationExample(
                id=message.id, text=message.text_clean, label=message.discourse_act,
                metadata={"conversation_id": conversation.id, "parent_text": parent_text},
            ))
    splits = split_by_group(examples, lambda ex: ex.metadata["conversation_id"], seed=seed)
    if not splits.train:
        raise ValueError("No training examples after filtering")
    unseen = {ex.label for ex in splits.dev + splits.test} - set(splits.labels)
    if unseen:
        raise ValueError(f"Labels absent from training: {sorted(unseen)}")
    return splits


def split_manifest(splits: ClassificationSplits, seed: int = 42) -> dict:
    """Fingerprint labels, child/parent text, and exact split membership."""
    rows = []
    for name in ("train", "dev", "test"):
        for ex in getattr(splits, name):
            payload = json.dumps([ex.text, ex.label, ex.metadata.get("parent_text", "")],
                                 ensure_ascii=False).encode()
            rows.append({"id": ex.id, "conversation_id": ex.metadata["conversation_id"],
                         "split": name, "sha256": hashlib.sha256(payload).hexdigest()})
    return {"version": 1, "seed": seed, "dev_frac": 0.1, "test_frac": 0.1,
            "labels": splits.labels, "examples": sorted(rows, key=lambda r: r["id"])}


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
