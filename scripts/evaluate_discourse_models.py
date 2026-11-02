"""Evaluate saved transformer and/or freshly trained TF-IDF baselines.

Defaults to dev. Explicitly pass --split test only after model selection.
Example:
    python scripts/evaluate_discourse_models.py --data DATA.jsonl --run data/local/discourse/plain --baselines --output data/local/discourse/eval-dev
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))
from discourse_data import load_discourse_data, save_json, split_manifest
from evaluation import classification_metrics


def export_result(output, name, examples, predictions, elapsed, sample_size, seed):
    result = classification_metrics([ex.label for ex in examples], predictions)
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "metrics.json", result.model_dump())
    normalized = [[value / sum(row) if sum(row) else 0.0 for value in row]
                  for row in result.confusion]
    save_json(folder / "normalized_confusion.json", {
        "labels": result.labels, "rows": "gold", "columns": "predicted", "matrix": normalized})
    for filename, matrix in (("confusion.csv", result.confusion),
                             ("normalized_confusion.csv", normalized)):
        with (folder / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gold / predicted", *result.labels])
            writer.writerows([label, *row] for label, row in zip(result.labels, matrix))
    errors = []
    with (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for ex, prediction in zip(examples, predictions):
            row = {"id": ex.id, "conversation_id": ex.metadata["conversation_id"],
                   "text": ex.text, "parent_text": ex.metadata.get("parent_text", ""),
                   "gold": ex.label, "prediction": prediction}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if ex.label != prediction:
                errors.append(row)
    # Stratify by gold class so majority-class errors do not swamp the review.
    rng = random.Random(seed)
    sample = []
    for label in result.labels:
        pool = [row for row in errors if row["gold"] == label]
        sample.extend(rng.sample(pool, min(sample_size, len(pool))))
    save_json(folder / "misclassification_sample.json", sample)
    save_json(folder / "inference_timing.json", {"seconds": elapsed,
              "examples_per_second": len(examples) / elapsed if elapsed else None})
    return {"model": name, "accuracy": result.accuracy,
            "macro_f1": result.macro_f1, "weighted_f1": result.weighted_f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="Training run directory containing best/ and split_manifest.json")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-per-class", type=int, default=10)
    args = parser.parse_args()
    if not args.run and not args.baselines:
        parser.error("Choose --run and/or --baselines")
    if args.batch_size < 1 or args.sample_per_class < 0:
        parser.error("Batch size must be positive and sample size nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    splits = load_discourse_data(args.data, args.split_seed)
    manifest = split_manifest(splits, args.split_seed)
    if args.run and json.loads((args.run / "split_manifest.json").read_text()) != manifest:
        parser.error("Data or split differs from the training run's manifest")
    examples = getattr(splits, args.split)
    if not examples:
        parser.error(f"No examples in {args.split}")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "split_manifest.json", manifest)
    save_json(args.output / "evaluation_config.json", {
        "split": args.split, "split_seed": args.split_seed, "n_examples": len(examples),
        "run": str(args.run) if args.run else None})
    results = []

    def evaluate(name, predict):
        started = time.perf_counter()
        predictions = predict()
        elapsed = time.perf_counter() - started
        results.append(export_result(args.output, name, examples, predictions, elapsed,
                                     args.sample_per_class, args.split_seed))

    if args.baselines:
        from collections import Counter
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        majority = Counter(ex.label for ex in splits.train).most_common(1)[0][0]
        evaluate("majority", lambda: [majority] * len(examples))
        for name, weight in (("tfidf_logreg", None), ("tfidf_balanced_logreg", "balanced")):
            model = Pipeline([
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                          max_features=100_000)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight=weight)),
            ])
            model.fit([ex.text for ex in splits.train], [ex.label for ex in splits.train])
            evaluate(name, lambda: model.predict([ex.text for ex in examples]).tolist())
    if args.run:
        from discourse_transformer import DiscourseTransformer
        model = DiscourseTransformer(args.run / "best")
        evaluate("transformer", lambda: model.predict_examples(examples, args.batch_size))
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "accuracy", "macro_f1", "weighted_f1"])
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
