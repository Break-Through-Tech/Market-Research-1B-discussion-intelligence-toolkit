"""Train and evaluate the lightweight quality scorer (November Task #1 / issue #29).

    python scripts/train_quality_scorer.py

1. Loads the LLM labels (`data/local/quality_labels/llm_labels-<model>.csv`) and
   joins them to the reply text.
2. Picks the ridge strength by cross-validation on the LLM labels only, grouped
   by CGA-CMV pair so a conversation and its match never straddle folds. The
   human-reviewed 60 are never used for tuning.
3. Trains on all LLM labels, saves the model to data/local/quality_scorer.joblib.
4. Scores the human-reviewed 60 (none of them are in the LLM training set) and
   reports MAE and Spearman per dimension, next to the same numbers for the LLM
   labels themselves.

Outputs go to reports/quality-labels/scorer/ and hold IDs and scores only, no
comment text.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts"))
from quality_labeler import DIMENSIONS, load_reply_rows  # noqa: E402
from quality_scorer import QualityScorer, evaluate_scores, large_errors  # noqa: E402

DEFAULT_DATA = ROOT / "data/local/standardized-v1/conversations-gone-awry-cmv-corpus.jsonl"
DEFAULT_LLM = ROOT / "data/local/quality_labels/llm_labels-claude-opus-5-5.csv"
DEFAULT_HUMAN = ROOT / "reports/quality-labels/human_reviewed_60.csv"
DEFAULT_LLM_EVAL = ROOT / "reports/quality-labels/eval-claude-opus-5-5-low-rubric-v1/llm_labels.csv"
DEFAULT_MODEL_OUT = ROOT / "data/local/quality_scorer.joblib"
DEFAULT_OUT = ROOT / "reports/quality-labels/scorer"
ALPHAS = (1.0, 3.0, 10.0)


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def cross_validate(train: pd.DataFrame, alpha: float, folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold scores on the LLM labels, grouped by pair."""
    groups = train["pair_id"].where(train["pair_id"] != "", train["conversation_id"])
    parts = []
    for fit_idx, test_idx in GroupKFold(n_splits=folds).split(train, groups=groups):
        model = QualityScorer(alpha=alpha).fit(train.iloc[fit_idx])
        scored = model.score(train.iloc[test_idx])
        scored["utterance_id"] = train.iloc[test_idx]["utterance_id"].to_numpy()
        parts.append(scored)
    oof = pd.concat(parts)
    return evaluate_scores(train, oof), oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--llm-labels", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--human-labels", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--llm-eval", type=Path, default=DEFAULT_LLM_EVAL,
                        help="LLM labels for the human-reviewed set, for side-by-side comparison")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = load_reply_rows(args.data)
    llm = read_csv(args.llm_labels)
    llm = llm[llm["status"] == "ok"].drop_duplicates("utterance_id", keep="last")
    human = read_csv(args.human_labels)
    overlap = set(llm["utterance_id"]) & set(human["utterance_id"])
    if overlap:
        sys.exit(f"{len(overlap)} human-reviewed replies are in the LLM training labels; they must be held out")

    text = rows[["utterance_id", "thread_start", "parent", "reply"]]
    train = llm[["utterance_id", "conversation_id", "pair_id", *DIMENSIONS]].merge(text, on="utterance_id")
    test = human.merge(text, on="utterance_id")
    print(f"training on {len(train)} LLM-labeled replies from {train['conversation_id'].nunique()} conversations; "
          f"testing on {len(test)} human-reviewed replies")

    # 1. choose alpha by cross-validation on LLM labels only
    cv_tables = {}
    for alpha in ALPHAS:
        table, _ = cross_validate(train, alpha)
        cv_tables[alpha] = table
        print(f"alpha={alpha}: CV MAE {table['mae'].mean():.3f}, Spearman {table['spearman'].mean():.3f}")
    best_alpha = min(ALPHAS, key=lambda a: cv_tables[a]["mae"].mean())
    cv_table = cv_tables[best_alpha]

    # 2. final model on all LLM labels
    scorer = QualityScorer(alpha=best_alpha).fit(train)
    scorer.save(args.model_out)

    # 3. held-out comparison with human labels
    scored = scorer.score(test)
    scored.insert(0, "utterance_id", test["utterance_id"].to_numpy())
    vs_human = evaluate_scores(human, scored)
    tables = {"scorer_vs_human": vs_human, "scorer_cv_vs_llm": cv_table}
    if args.llm_eval.exists():
        tables["llm_vs_human"] = evaluate_scores(human, read_csv(args.llm_eval))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
    scored.round(3).to_csv(args.out_dir / "scorer_scores_human60.csv", index=False)
    errors = large_errors(human, scored)
    keep = ["utterance_id", "sample_group", *[c for d in DIMENSIONS for c in (d, f"{d}_score", f"{d}_error")], "note"]
    errors[keep].round(3).to_csv(args.out_dir / "scorer_errors.csv", index=False)

    summary = {
        "date": date.today().isoformat(),
        "llm_labels": str(args.llm_labels.relative_to(ROOT)) if args.llm_labels.is_relative_to(ROOT) else str(args.llm_labels),
        "n_train": len(train),
        "n_train_conversations": int(train["conversation_id"].nunique()),
        "n_test_human": len(test),
        "alpha": best_alpha,
        "cv_mae_by_alpha": {str(a): float(t["mae"].mean()) for a, t in cv_tables.items()},
        "model_path": str(args.model_out.relative_to(ROOT)) if args.model_out.is_relative_to(ROOT) else str(args.model_out),
        **{name: table.to_dict("records") for name, table in tables.items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fmt = lambda x: f"{x:.2f}"  # noqa: E731
    for name, table in tables.items():
        print(f"\n{name}\n{table.to_string(index=False, float_format=fmt)}")
    print(f"\n{len(errors)} of {len(test)} human-reviewed replies have an error >= 0.75 on some dimension")
    print("model saved to", args.model_out, "| reports in", args.out_dir)


if __name__ == "__main__":
    main()
