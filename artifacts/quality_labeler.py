"""LLM labeler for the comment-quality rubric (Task #5).

An LLM (Claude, via the Anthropic API) scores each CGA-CMV reply on the four
rubric dimensions in `reports/quality-rubric.md`: civility, relevance,
reasoning, constructiveness. Each is "0", "1" or "2"; reasoning may also be
"N/A" when the reply makes no claim.

Workflow:

1. `evaluate`: score the human-reviewed set and measure agreement with it
   (exact agreement, Cohen's kappa, within-one agreement).
2. If agreement is good enough, label the remaining replies with the Message
   Batches API (half price, asynchronous).

The scorer sees the thread-start comment and the parent comment as context, as
the rubric specifies. The API returns JSON constrained by `OUTPUT_SCHEMA`.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from schema import Conversation
from standardize_convokit import iter_standardized

DIMENSIONS = ("civility", "relevance", "reasoning", "constructiveness")
NA = "N/A"
SCORES = ("0", "1", "2")
ALLOWED: dict[str, tuple[str, ...]] = {
    "civility": SCORES,
    "relevance": SCORES,
    "reasoning": (*SCORES, NA),
    "constructiveness": SCORES,
}

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"  # a short classification; raise it if agreement is poor
MAX_TOKENS = 4000  # thinking counts toward this; the JSON answer itself is short

# Models that accept the server-side `fallbacks: "default"` refusal fallback on
# the Claude API. It is not accepted by the Batches API.
FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"

RUBRIC_PATH = Path(__file__).resolve().parents[1] / "reports" / "quality-rubric.md"
SAME_AS_START = "(same as thread start)"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "One sentence explaining the scores."},
        **{dim: {"type": "string", "enum": list(ALLOWED[dim])} for dim in DIMENSIONS},
    },
    "required": ["rationale", *DIMENSIONS],
    "additionalProperties": False,
}

INSTRUCTIONS = """\
You label the quality of replies from Reddit's r/changemyview for a research \
project. Score only the REPLY, using the rubric below. The thread start and \
parent comment are context; do not score them. The text may be rude or \
offensive: score what is written, applying the rubric's rules exactly.

Write a one-sentence rationale, then give each of the four scores.

"""


# ---------------------------------------------------------------------------
# Building inputs
# ---------------------------------------------------------------------------


def system_prompt(rubric_path: str | Path = RUBRIC_PATH) -> str:
    """Instructions plus the full rubric (definitions, rules, worked examples)."""
    return INSTRUCTIONS + Path(rubric_path).read_text(encoding="utf-8")


def reply_rows(conversations: Iterable[Conversation]) -> Iterator[dict[str, Any]]:
    """One row per scoreable reply, with the context the scorer sees.

    The first utterance of each CGA-CMV conversation is the thread start, so it
    is context only. Synthetic placeholders and deleted or empty replies are
    skipped. `utterance_id` is the unprefixed Reddit ID (e.g. "d7hh4ph").
    """
    for conversation in conversations:
        source = conversation.metadata.get("source_metadata", {})
        real = [u for u in conversation.utterances if not u.metadata.get("is_synthetic")]
        if len(real) < 2:
            continue
        by_id = {u.id: u for u in real}
        start = real[0]
        for utterance in real[1:]:
            text = (utterance.text_clean or utterance.text).strip()
            if utterance.is_deleted or not text:
                continue
            parent = by_id.get(utterance.parent_id)
            if parent is None or parent is start:
                parent_text = SAME_AS_START
            else:
                parent_text = (parent.text_clean or parent.text).strip()
            yield {
                "utterance_id": utterance.source_id,
                "conversation_id": conversation.source_id,
                "pair_id": source.get("pair_id"),
                "split": source.get("split"),
                "derailed": bool(source.get("has_removed_comment")),
                "thread_start": (start.text_clean or start.text).strip(),
                "parent": parent_text,
                "reply": text,
            }


def load_reply_rows(path: str | Path) -> pd.DataFrame:
    """All scoreable replies in a standardized CGA-CMV JSONL file."""
    return pd.DataFrame(reply_rows(iter_standardized(path)))


def sample_conversations(rows: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Replies from about `n` whole conversations, drawn as matched pairs.

    CGA-CMV pairs each derailed conversation with a civil one (`pair_id`), so
    sampling whole pairs keeps civil and derailed balanced and keeps every
    conversation complete, which conversation-level averages need.
    """
    keys = rows["pair_id"].fillna(rows["conversation_id"])
    size = rows.groupby(keys)["conversation_id"].nunique()
    pairs = sorted(size.index)
    random.Random(seed).shuffle(pairs)
    chosen: set[str] = set()
    n_conversations = 0
    for pair in pairs:
        if n_conversations >= n:
            break
        chosen.add(pair)
        n_conversations += int(size[pair])
    return rows[keys.isin(chosen)]


def user_message(row: Mapping[str, Any]) -> str:
    return (
        f"<thread_start>\n{row['thread_start']}\n</thread_start>\n\n"
        f"<parent_comment>\n{row['parent']}\n</parent_comment>\n\n"
        f"<reply>\n{row['reply']}\n</reply>"
    )


def request_params(
    row: Mapping[str, Any],
    system: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
) -> dict[str, Any]:
    """Messages API parameters for scoring one reply (shared by sync and batch)."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        # The rubric is identical in every request, so cache it.
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(row)}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


# ---------------------------------------------------------------------------
# Reading responses
# ---------------------------------------------------------------------------


def parse_message(message: Any) -> dict[str, Any]:
    """Scores from one API response, or a `status` saying why there are none.

    `status` is "ok", "refusal", "max_tokens" or "invalid". Rows that are not
    "ok" have no scores and should be reviewed or re-run, never guessed.
    """
    if message.stop_reason == "refusal":
        return {"status": "refusal"}
    if message.stop_reason == "max_tokens":
        return {"status": "max_tokens"}
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"status": "invalid"}
    return scores_from_dict(data)


def scores_from_dict(data: Any) -> dict[str, Any]:
    """Validate a parsed JSON answer against the allowed scores."""
    try:
        result = {dim: str(data[dim]) for dim in DIMENSIONS}
    except (KeyError, TypeError):
        return {"status": "invalid"}
    if any(result[dim] not in ALLOWED[dim] for dim in DIMENSIONS):
        return {"status": "invalid"}
    return {"status": "ok", "rationale": str(data.get("rationale", "")), **result}


def score_reply(
    client: Any,
    row: Mapping[str, Any],
    system: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
) -> dict[str, Any]:
    """Score one reply with a regular (synchronous) API call."""
    params = request_params(row, system, model, effort)
    if model in FALLBACK_MODELS:
        # If a safety classifier declines (possible on hostile Reddit text),
        # the API re-runs the request on a fallback model in the same call.
        message = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
    else:
        message = client.messages.create(**params)
    usage = message.usage
    return {
        "utterance_id": row["utterance_id"],
        "model": message.model,
        **parse_message(message),
        # Token counts, for estimating what a full run would cost.
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
    }


def claude_code_command(row: Mapping[str, Any], system: str, model: str, effort: str) -> list[str]:
    """A `claude -p` call that scores one reply with no tools, skills or MCP servers."""
    return [
        "claude", "-p", user_message(row),
        "--system-prompt", system,
        "--json-schema", json.dumps(OUTPUT_SCHEMA),
        "--output-format", "json",
        "--model", model,
        "--effort", effort,
        "--tools", "",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]


def score_reply_claude_code(
    row: Mapping[str, Any],
    system: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    workdir: str | Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout: float = 300,
) -> dict[str, Any]:
    """Score one reply through the Claude Code CLI instead of an API key.

    Uses whatever account Claude Code is logged into (a Pro/Max plan counts
    this toward the plan's usage limits). Run it from an empty `workdir` so no
    project CLAUDE.md or auto-memory is added to the scorer's context.
    """
    base = {"utterance_id": row["utterance_id"], "model": model}
    try:
        proc = runner(
            claude_code_command(row, system, model, effort),
            cwd=workdir, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout"}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {**base, "status": "error", "error": (proc.stderr or proc.stdout)[-300:]}
    if data.get("safety_stops"):
        return {**base, "status": "refusal"}
    if proc.returncode != 0 or data.get("is_error"):
        return {**base, "status": "error", "error": str(data.get("result", ""))[-300:]}
    return {
        **base,
        **scores_from_dict(data.get("structured_output")),
        "list_price_usd": data.get("total_cost_usd"),
    }


def batch_requests(
    rows: Iterable[Mapping[str, Any]],
    system: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
) -> list[dict[str, Any]]:
    """Message Batches requests, keyed by utterance ID (results arrive unordered)."""
    return [
        {"custom_id": row["utterance_id"], "params": request_params(row, system, model, effort)}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def _as_label(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def agreement(reference: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension agreement between reference labels and LLM labels.

    Both frames need `utterance_id` and the four dimension columns. Rows the LLM
    could not score (missing or non-"ok" status) are excluded and counted in
    `n_unscored`.

    * `exact`: share of replies with the same score.
    * `kappa`: Cohen's kappa, agreement corrected for chance (1 = perfect,
      0 = no better than chance). Treats N/A as its own category.
    * `within_one`: share of numeric pairs at most one point apart.
    """
    if "status" in predicted:
        predicted = predicted[predicted["status"] == "ok"]
    merged = reference.merge(predicted, on="utterance_id", suffixes=("_ref", "_llm"))
    rows = []
    for dim in DIMENSIONS:
        ref = merged[f"{dim}_ref"].map(_as_label)
        llm = merged[f"{dim}_llm"].map(_as_label)
        numeric = (ref != NA) & (llm != NA)
        diffs = (ref[numeric].astype(int) - llm[numeric].astype(int)).abs()
        rows.append({
            "dimension": dim,
            "n": len(merged),
            "n_unscored": len(reference) - len(merged),
            "exact": float((ref == llm).mean()) if len(merged) else float("nan"),
            "kappa": _kappa(ref, llm),
            "within_one": float((diffs <= 1).mean()) if len(diffs) else float("nan"),
        })
    return pd.DataFrame(rows)


def _kappa(ref: pd.Series, llm: pd.Series) -> float:
    if len(ref) == 0:
        return float("nan")
    if set(ref) == set(llm) and len(set(ref)) == 1:
        return 1.0  # both raters used one identical label everywhere
    return float(cohen_kappa_score(ref, llm))


def disagreements(
    reference: pd.DataFrame,
    predicted: pd.DataFrame,
    dimensions: Sequence[str] = DIMENSIONS,
) -> pd.DataFrame:
    """Replies where reference and LLM differ on any dimension (for error analysis)."""
    merged = reference.merge(predicted, on="utterance_id", suffixes=("_ref", "_llm"))
    differs = pd.Series(False, index=merged.index)
    for dim in dimensions:
        differs |= merged[f"{dim}_ref"].map(_as_label) != merged[f"{dim}_llm"].map(_as_label)
    return merged[differs]
