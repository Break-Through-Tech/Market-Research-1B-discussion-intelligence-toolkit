"""Label CGA-CMV replies with the quality rubric using Claude (Task #5).

Two ways to reach Claude (`--backend`):

* `claude-code` (default): the Claude Code CLI, using the account it is logged
  into. No API key needed; on a Pro/Max plan the calls count toward the plan's
  usage limits. Slower (a few seconds per reply), so it runs several at once.
* `api`: the Anthropic API with ANTHROPIC_API_KEY, billed per token. Also
  allows the Message Batches API (`submit` / `collect`, half price).

1. Check agreement with the human-reviewed labels (60 calls):
    python scripts/label_quality.py evaluate --model claude-haiku-5-5

2. Label a sample of whole conversations (resumable: re-run to continue):
    python scripts/label_quality.py label --conversations 1000 --dry-run
    python scripts/label_quality.py label --conversations 1000

   Or, with an API key, as batches:
    python scripts/label_quality.py submit --conversations 1000
    python scripts/label_quality.py collect   # re-run until all batches end

Labels go to data/local/quality_labels/ (gitignored). Evaluation outputs go to
reports/quality-labels/ and hold IDs, scores and LLM rationales, no comment text.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts"))
from quality_labeler import (  # noqa: E402
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    DIMENSIONS,
    agreement,
    batch_requests,
    disagreements,
    load_reply_rows,
    parse_message,
    sample_conversations,
    score_reply,
    score_reply_claude_code,
    system_prompt,
)

DEFAULT_DATA = ROOT / "data/local/standardized-v1/conversations-gone-awry-cmv-corpus.jsonl"
DEFAULT_LABELS = ROOT / "reports/quality-labels/human_reviewed_60.csv"
LABEL_DIR = ROOT / "data/local/quality_labels"
BATCH_LOG = LABEL_DIR / "batches.json"
MAX_FAILURES_IN_A_ROW = 10


def read_csv(path) -> pd.DataFrame:
    """Read labels as text. Pandas would otherwise turn the score "N/A" into a missing value."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def make_scorer(args: argparse.Namespace):
    """A function row -> result dict for the chosen backend."""
    system = system_prompt()
    if args.backend == "api":
        import anthropic

        client = anthropic.Anthropic()
        return lambda row: score_reply(client, row, system, args.model, args.effort)
    workdir = tempfile.mkdtemp(prefix="quality-labeler-")  # empty: no CLAUDE.md or memory
    return lambda row: score_reply_claude_code(row, system, args.model, args.effort, workdir)


def score_rows(rows: pd.DataFrame, args: argparse.Namespace, on_result=None) -> list[dict]:
    """Score rows with `args.workers` calls in flight, reporting progress."""
    scorer = make_scorer(args)
    results, start, failures_in_a_row = [], time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scorer, row) for row in rows.to_dict("records")]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if on_result:
                on_result(result)
            failures_in_a_row = failures_in_a_row + 1 if result["status"] in ("error", "timeout") else 0
            if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                # Usually a usage limit or a lost login: stop rather than fail every remaining call.
                pool.shutdown(cancel_futures=True)
                print(f"\nStopping: {failures_in_a_row} failures in a row. Last error: "
                      f"{result.get('error', result['status'])}")
                break
            rate = (time.time() - start) / i
            print(f"\r{i}/{len(rows)} scored, ~{rate * (len(rows) - i) / 60:.0f} min left",
                  end="", flush=True)
    print()
    return results


def choose_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = load_reply_rows(args.data)
    if args.exclude:
        rows = rows[~rows["utterance_id"].isin(read_csv(args.exclude)["utterance_id"])]
    if args.split != "all":
        rows = rows[rows["split"] == args.split]
    if args.conversations:
        rows = sample_conversations(rows, args.conversations, seed=args.seed)
    print(f"{len(rows)} replies from {rows['conversation_id'].nunique()} conversations "
          f"({rows.drop_duplicates('conversation_id')['derailed'].sum()} derailed)")
    return rows


def evaluate(args: argparse.Namespace) -> None:
    labels = read_csv(args.labels)
    rows = load_reply_rows(args.data)
    rows = rows[rows["utterance_id"].isin(labels["utterance_id"])]
    missing = set(labels["utterance_id"]) - set(rows["utterance_id"])
    if missing:
        sys.exit(f"{len(missing)} labeled replies not found in {args.data}: {sorted(missing)[:5]}")

    out = Path(args.out_dir or ROOT / f"reports/quality-labels/eval-{args.model}-{args.effort}")
    if args.reuse:  # re-analyze saved scores without calling Claude again
        llm = read_csv(out / "llm_labels.csv")
        for column in llm.columns:
            if column.endswith("_tokens") or column == "list_price_usd":
                llm[column] = pd.to_numeric(llm[column])
    else:
        llm = pd.DataFrame(score_rows(rows, args))
        out.mkdir(parents=True, exist_ok=True)
        llm.to_csv(out / "llm_labels.csv", index=False)
    table = agreement(labels, llm)
    table.to_csv(out / "agreement.csv", index=False)
    keep = ["utterance_id", *[f"{d}_{s}" for d in DIMENSIONS for s in ("ref", "llm")], "note", "rationale"]
    disagreements(labels, llm[llm["status"] == "ok"])[keep].to_csv(out / "disagreements.csv", index=False)
    numeric = [c for c in llm.columns if c.endswith("_tokens") or c == "list_price_usd"]
    summary = {
        "model": args.model,
        "effort": args.effort,
        "backend": args.backend,
        "n_labels": len(labels),
        "status_counts": llm["status"].value_counts().to_dict(),
        "usage_totals": {c: float(llm[c].fillna(0).sum()) for c in numeric},
        "agreement": table.to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("status:", summary["status_counts"])
    print("usage:", summary["usage_totals"])
    print("saved to", out)


def label(args: argparse.Namespace) -> None:
    rows = choose_rows(args)
    out = Path(args.out or LABEL_DIR / f"llm_labels-{args.model}.csv")
    done = set()
    if out.exists():
        previous = read_csv(out)
        done = set(previous.loc[previous["status"] == "ok", "utterance_id"])
    rows = rows[~rows["utterance_id"].isin(done)]
    print(f"{len(done)} already labeled in {out}; {len(rows)} to go")
    if args.dry_run or rows.empty:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    meta = rows.set_index("utterance_id")[["conversation_id", "pair_id", "split", "derailed"]]

    def append(result: dict) -> None:  # save as we go, so an interruption loses nothing
        record = {**meta.loc[result["utterance_id"]].to_dict(), **result}
        pd.DataFrame([record]).to_csv(out, mode="a", header=not out.exists(), index=False)

    results = pd.DataFrame(score_rows(rows, args, on_result=append))
    print(results["status"].value_counts().to_string())
    print("saved to", out, "(re-run the same command to retry anything not 'ok')")


def submit(args: argparse.Namespace) -> None:
    rows = choose_rows(args)
    chunks = [rows.iloc[i:i + args.chunk_size] for i in range(0, len(rows), args.chunk_size)]
    print(f"{len(chunks)} batch(es) of up to {args.chunk_size}")
    if args.dry_run:
        return

    import anthropic

    client, system = anthropic.Anthropic(), system_prompt()
    log = json.loads(BATCH_LOG.read_text()) if BATCH_LOG.exists() else []
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        batch = client.messages.batches.create(
            requests=batch_requests(chunk.to_dict("records"), system, args.model, args.effort)
        )
        log.append({"id": batch.id, "n": len(chunk), "model": args.model, "effort": args.effort,
                    "submitted": time.strftime("%Y-%m-%d %H:%M:%S")})
        BATCH_LOG.write_text(json.dumps(log, indent=2))  # save after each, in case one fails
        print("submitted", batch.id, len(chunk))


def collect(args: argparse.Namespace) -> None:
    if not BATCH_LOG.exists():
        sys.exit(f"No batches recorded in {BATCH_LOG}; run `submit` first.")
    import anthropic

    client = anthropic.Anthropic()
    log = json.loads(BATCH_LOG.read_text())
    pending = []
    for entry in log:
        batch = client.messages.batches.retrieve(entry["id"])
        print(entry["id"], batch.processing_status, dict(batch.request_counts))
        if batch.processing_status != "ended":
            pending.append(entry["id"])
    if pending:
        print(f"{len(pending)} batch(es) still running; run `collect` again later.")
        return

    results = []
    for entry in log:
        for item in client.messages.batches.results(entry["id"]):
            row = {"utterance_id": item.custom_id, "model": entry["model"], "batch_id": entry["id"]}
            if item.result.type == "succeeded":
                row.update(parse_message(item.result.message))
            else:  # errored / canceled / expired: re-submit these
                row["status"] = item.result.type
            results.append(row)
    out = LABEL_DIR / "llm_labels-batch.csv"
    frame = pd.DataFrame(results).drop_duplicates("utterance_id", keep="last")
    frame.to_csv(out, index=False)
    print(frame["status"].value_counts().to_string())
    print("saved", len(frame), "labels to", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="standardized CGA-CMV JSONL")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(p, backend=True):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
        if backend:
            p.add_argument("--backend", default="claude-code", choices=["claude-code", "api"])
            p.add_argument("--workers", type=int, default=4, help="calls in flight at once")

    def selection_args(p):
        p.add_argument("--conversations", type=int, help="sample about N whole conversations (as matched pairs)")
        p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
        p.add_argument("--exclude", type=Path, default=DEFAULT_LABELS, help="CSV of utterance_ids to skip")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--dry-run", action="store_true", help="count replies without calling Claude")

    p = sub.add_parser("evaluate", help="score the human-reviewed set and report agreement")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--reuse", action="store_true", help="re-analyze saved llm_labels.csv, no new calls")
    model_args(p)
    p.set_defaults(func=evaluate)

    p = sub.add_parser("label", help="label replies one call at a time (resumable)")
    p.add_argument("--out", type=Path)
    selection_args(p)
    model_args(p)
    p.set_defaults(func=label)

    p = sub.add_parser("submit", help="submit Message Batches (API key required)")
    p.add_argument("--chunk-size", type=int, default=10_000, help="requests per batch (size limit is 256 MB)")
    selection_args(p)
    model_args(p, backend=False)
    p.set_defaults(func=submit)

    p = sub.add_parser("collect", help="download finished batch results (API key required)")
    p.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
