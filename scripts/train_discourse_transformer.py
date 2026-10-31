"""Train on standardized JSONL; checkpoint selection uses dev macro F1.

Example (install torch, transformers, scikit-learn, scipy, numpy, pydantic first):
    python scripts/train_discourse_transformer.py --data DATA.jsonl --output data/local/discourse/plain
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))
from discourse_data import load_discourse_data, save_json, split_manifest
from discourse_transformer import train_transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New/empty run directory; use data/local/ to keep artifacts untracked")
    parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42, help="Model random seed")
    parser.add_argument("--split-seed", type=int, default=42, help="Keep fixed across experiments")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--parent-context", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; use a different directory for each experiment")
    splits = load_discourse_data(args.data, args.split_seed)
    print(splits.summary(), flush=True)
    try:
        train_transformer(splits, args.output, model_name=args.model_name, seed=args.seed,
                          epochs=args.epochs, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, max_length=args.max_length,
                          weighted=args.weighted, parent_context=args.parent_context,
                          accumulation_steps=args.accumulation_steps,
                          mixed_precision=not args.no_mixed_precision)
    except Exception:
        save_json(args.output / "failure.json", {"arguments": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, "traceback": traceback.format_exc()})
        raise
    finally:
        save_json(args.output / "split_manifest.json", split_manifest(splits, args.split_seed))


if __name__ == "__main__":
    main()
