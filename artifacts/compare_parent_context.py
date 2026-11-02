"""Compare PR #21's saved discourse predictions without retraining.

Pass evaluation folders (plain_test and parent_context_test), each containing
evaluation_config.json, split_manifest.json, transformer/metrics.json, and
transformer/predictions.jsonl.
Run with --help for the complete command. Full predictions remain local.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re

from evaluation import classification_metrics

FIELDS = ("id", "conversation_id", "text", "parent_text", "gold", "prediction")
MATCH_FIELDS = ("conversation_id", "text", "parent_text", "gold")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(row: dict) -> str:
    """Match discourse_data.split_manifest's exact UTF-8 payload encoding."""
    payload = json.dumps([row["text"], row["gold"], row["parent_text"]], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_evaluation(folder: Path) -> dict:
    """Validate an export against its full split manifest and stored metrics."""
    config = read_json(folder / "evaluation_config.json")
    manifest = read_json(folder / "split_manifest.json")
    split = config["split"]
    if split not in ("dev", "test") or config["split_seed"] != manifest["seed"]:
        raise ValueError("Invalid evaluation split or inconsistent split seed")
    if manifest["version"] != 1:
        raise ValueError("Unsupported split manifest version")
    labels = manifest["labels"]
    if not labels or len(set(labels)) != len(labels) or not all(isinstance(x, str) and x for x in labels):
        raise ValueError("Invalid manifest label inventory")
    expected, groups = {}, {}
    for row in manifest["examples"]:
        uid, cid, bucket = row["id"], row["conversation_id"], row["split"]
        if uid in expected:
            raise ValueError(f"Duplicate manifest ID: {uid}")
        if not uid or not cid or bucket not in ("train", "dev", "test"):
            raise ValueError("Invalid manifest example")
        if cid in groups and groups[cid] != bucket:
            raise ValueError(f"Conversation crosses splits: {cid}")
        groups[cid] = bucket
        expected[uid] = row

    predictions = {}
    with (folder / "transformer" / "predictions.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in FIELDS):
                raise ValueError(f"Prediction line {number}: required string field missing")
            uid = row["id"]
            if uid in predictions:
                raise ValueError(f"Duplicate prediction ID: {uid}")
            if not uid or not row["text"].strip() or row["gold"] not in labels or row["prediction"] not in labels:
                raise ValueError(f"Invalid prediction record: {uid}")
            entry = expected.get(uid)
            if (entry is None or entry["split"] != split
                    or entry["conversation_id"] != row["conversation_id"]
                    or entry["sha256"] != fingerprint(row)):
                raise ValueError(f"Prediction does not match split manifest: {uid}")
            predictions[uid] = row
    expected_ids = {uid for uid, row in expected.items() if row["split"] == split}
    if not predictions or set(predictions) != expected_ids or len(predictions) != config["n_examples"]:
        raise ValueError("Prediction population differs from evaluation config or manifest")
    ordered = [predictions[uid] for uid in sorted(predictions)]
    metrics = classification_metrics([row["gold"] for row in ordered],
                                     [row["prediction"] for row in ordered]).model_dump()
    saved = read_json(folder / "transformer" / "metrics.json")
    if metrics["n_examples"] != saved["n_examples"] or metrics["labels"] != saved["labels"]:
        raise ValueError("Stored metrics have different examples or labels")
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        if not math.isclose(metrics[key], saved[key], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Stored {key} does not match exported predictions")
    if (metrics["confusion"] != saved["confusion"]
            or set(metrics["per_class_f1"]) != set(saved["per_class_f1"])
            or any(not math.isclose(value, saved["per_class_f1"][label], rel_tol=1e-9, abs_tol=1e-12)
                   for label, value in metrics["per_class_f1"].items())):
        raise ValueError("Stored per-class metrics or confusion do not match exported predictions")
    return {"config": config, "manifest": manifest, "predictions": predictions, "metrics": metrics}


def align_evaluations(plain: dict, context: dict) -> list[dict]:
    for key in ("split", "split_seed", "n_examples"):
        if plain["config"][key] != context["config"][key]:
            raise ValueError(f"Evaluations have different {key}")
    # The upstream writer sorts manifest rows by ID. Normalize that order as
    # well, so file serialization order cannot change pairing or comparability.
    def normalized(manifest):
        return {**manifest, "examples": sorted(manifest["examples"], key=lambda row: row["id"])}
    if normalized(plain["manifest"]) != normalized(context["manifest"]):
        raise ValueError("Evaluations have different split manifests")
    if set(plain["predictions"]) != set(context["predictions"]):
        raise ValueError("Evaluations have different prediction IDs")
    if plain["metrics"]["labels"] != context["metrics"]["labels"]:
        raise ValueError("Evaluations have different metric label inventories")
    rows = []
    for uid in sorted(plain["predictions"]):
        before, after = plain["predictions"][uid], context["predictions"][uid]
        if any(before[key] != after[key] for key in MATCH_FIELDS):
            raise ValueError(f"Gold label or source text differs for {uid}")
        correct_before = before["prediction"] == before["gold"]
        correct_after = after["prediction"] == after["gold"]
        effect = ("both_correct" if correct_before and correct_after else
                  "helps" if correct_after else "hurts" if correct_before else "both_wrong")
        rows.append({**{key: before[key] for key in ("id", *MATCH_FIELDS)},
                     "comment_only": before["prediction"], "parent_context": after["prediction"],
                     "effect": effect, "has_parent_text": bool(before["parent_text"].strip())})
    return rows


def transition_summary(rows: list[dict]) -> dict:
    effects = Counter(row["effect"] for row in rows)
    n = len(rows)
    return {"n_examples": n,
            **{effect: effects[effect] for effect in ("helps", "hurts", "both_correct", "both_wrong")},
            "comment_only_accuracy": (effects["both_correct"] + effects["hurts"]) / n if n else None,
            "parent_context_accuracy": (effects["both_correct"] + effects["helps"]) / n if n else None}


def sample_changes(rows: list[dict], count: int = 3, seed: int = 42) -> list[dict]:
    """Select reproducible, class-diverse examples from each correctness change."""
    rng = random.Random(seed)
    sampled = []
    for effect in ("helps", "hurts"):
        by_label = {}
        for row in rows:
            if row["effect"] == effect:
                by_label.setdefault(row["gold"], []).append(row)
        labels = sorted(by_label)
        rng.shuffle(labels)
        for pool in by_label.values():
            rng.shuffle(pool)
        selected = []
        while labels and len(selected) < count:
            for label in labels[:]:
                selected.append(by_label[label].pop())
                if not by_label[label]:
                    labels.remove(label)
                if len(selected) == count:
                    break
        sampled.extend(selected)
    return sampled


def reddit_url(row: dict) -> str | None:
    prefix = "reddit:reddit-coarse-discourse-corpus:"
    if not row["id"].startswith(prefix + "utterance:") or not row["conversation_id"].startswith(prefix + "conversation:"):
        return None
    comment = re.sub(r"^t[13]_", "", row["id"].rsplit(":", 1)[-1])
    thread = re.sub(r"^t3_", "", row["conversation_id"].rsplit(":", 1)[-1])
    if not re.fullmatch(r"[a-z0-9]+", comment) or not re.fullmatch(r"[a-z0-9]+", thread):
        return None
    return f"https://www.reddit.com/comments/{thread}/" + (f"_/{comment}/" if comment != thread else "")


def render_report(summary: dict, per_class: list[dict], examples: list[dict]) -> str:
    lines = ["# Parent-context paired analysis", "",
             f"Split: **{summary['split']}**. Both models use the same **{summary['n_examples']:,}** examples.", "",
             "| Model | Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|"]
    for key, name in (("comment_only", "Comment only"), ("parent_context", "Parent context")):
        metrics = summary["metrics"][key]
        lines.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |")
    lines += ["", "## Per-class F1", "", "| Gold class | Support | Comment only | Parent context | Delta |",
              "|---|---:|---:|---:|---:|"]
    for row in per_class:
        lines.append(f"| {row['label']} | {row['support']} | {row['comment_only']:.4f} | {row['parent_context']:.4f} | {row['delta']:+.4f} |")
    lines += ["", "## Paired correctness changes", "",
              "Helps = wrong → correct; hurts = correct → wrong. Both-wrong includes changed predictions that remain wrong.", "",
              "Available means nonempty exported parent text, including deletion markers. Empty parent text includes roots and unavailable parents.", "",
              "| Parent text | Examples | Helps | Hurts | Both correct | Both wrong | Comment accuracy | Context accuracy |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, counts in summary["transitions"].items():
        accuracy = ["—" if counts[key] is None else f"{counts[key]:.4f}"
                    for key in ("comment_only_accuracy", "parent_context_accuracy")]
        lines.append(f"| {name} | {counts['n_examples']} | {counts['helps']} | {counts['hurts']} | "
                     f"{counts['both_correct']} | {counts['both_wrong']} | {' | '.join(accuracy)} |")
    lines += ["", "## Examples", "",
              f"Class-diverse samples use seed {summary['sample_seed']}. Text is excerpted to 400 characters; full exports remain local.",
              "These are correctness changes against corpus labels, not causal explanations. Live Reddit pages may differ or be unavailable."]
    for row in examples:
        url = reddit_url(row)
        reference = f"[Source]({url})" if url else f"Source ID: `{row['id']}`"
        lines += ["", f"### Context {row['effect']} — {row['gold']}", "", reference, "",
                  f"Comment only: **{row['comment_only']}**. Parent context: **{row['parent_context']}**.", ""]
        for key, name in (("parent_text", "Parent"), ("text", "Reply")):
            text = " ".join(row[key].split())
            excerpt = text[:400] + ("…" if len(text) > 400 else "")
            # Indented code blocks preserve arbitrary source Markdown safely.
            lines += [f"{name}:", "", "    " + (excerpt or "(empty parent text)"), ""]
    return "\n".join(lines).rstrip() + "\n"


def compare(plain_folder: Path, context_folder: Path, output: Path,
            examples_per_effect: int = 3, seed: int = 42) -> dict:
    if examples_per_effect < 0:
        raise ValueError("Example count must be nonnegative")
    plain, context = load_evaluation(plain_folder), load_evaluation(context_folder)
    rows = align_evaluations(plain, context)
    labels = plain["metrics"]["labels"]
    support = Counter(row["gold"] for row in rows)
    per_class = [{"label": label, "support": support[label],
                  "comment_only": plain["metrics"]["per_class_f1"][label],
                  "parent_context": context["metrics"]["per_class_f1"][label],
                  "delta": context["metrics"]["per_class_f1"][label] - plain["metrics"]["per_class_f1"][label]}
                 for label in labels]
    input_hashes = {}
    for name, folder in (("comment_only", plain_folder), ("parent_context", context_folder)):
        input_hashes[name] = {}
        for filename in ("evaluation_config.json", "split_manifest.json", "transformer/predictions.jsonl", "transformer/metrics.json"):
            with (folder / filename).open("rb") as stream:
                input_hashes[name][filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {"split": plain["config"]["split"], "split_seed": plain["config"]["split_seed"],
               "sample_seed": seed, "n_examples": len(rows), "input_sha256": input_hashes,
               "metrics": {"comment_only": plain["metrics"], "parent_context": context["metrics"]},
               "transitions": {"all": transition_summary(rows),
                               "available": transition_summary([row for row in rows if row["has_parent_text"]]),
                               "empty": transition_summary([row for row in rows if not row["has_parent_text"]])}}
    examples = sample_changes(rows, examples_per_effect, seed)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_class_f1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    (output / "report.md").write_text(render_report(summary, per_class, examples), encoding="utf-8")
    # Summary is written last as the completion marker. No full predictions are copied.
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-evaluation", type=Path, required=True)
    parser.add_argument("--context-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/local/parent-context-comparison"))
    parser.add_argument("--examples-per-effect", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Example sampling seed; does not change the evaluation split")
    args = parser.parse_args()
    try:
        result = compare(args.plain_evaluation, args.context_evaluation, args.output,
                         args.examples_per_effect, args.seed)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")
    print(f"Compared {result['n_examples']} {result['split']} examples; report written to {args.output}")


if __name__ == "__main__":
    main()
