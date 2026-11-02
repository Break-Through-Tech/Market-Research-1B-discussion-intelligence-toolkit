# October Task 3.5 review and paired analysis

PR #21 implements the core parent-comment context experiment. PR #23 added the implementation review and comparison tooling. The remaining empirical analysis is now complete using the original evaluation exports supplied by Jay. See the [full paired results](parent-context/README.md) for both models' per-class F1, correctness changes, reviewed examples, and reproduction instructions.

**Status: review, tooling, and paired analysis complete.** On 10,343 held-out test examples, the saved predictions reproduce macro F1 of 0.4996 for comment-only DistilBERT and 0.5766 with parent context. Context corrects 1,849 errors and introduces 868, giving 981 additional correct predictions. The analysis is ready for review under [issue #22](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/issues/22).

## Review of PR 21

Reviewed on November 1, 2026, at commit [`df106744b01854ce2b70065872ae6341b8b36b28`](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/pull/21/commits/df106744b01854ce2b70065872ae6341b8b36b28). The models, training runs, and original reported results belong to that PR. This follow-up verifies its saved predictions and completes the missing paired analysis. No training or inference was repeated.

| Task 3.5 requirement | Evidence | Coverage |
|---|---|---|
| Construct parent-reply inputs | PR #21 resolves the immediate parent within each standardized conversation | Covered |
| Compare comment-only and parent-context classifiers | Separate three-epoch DistilBERT experiments use the same examples and conversation-level split | Covered |
| Report macro F1 | Development and test scores recomputed from the original exports | Verified |
| Compare per-class F1 | Both models' F1, support, and deltas for all nine classes, on both splits | Completed in the paired results |
| Show examples where context helps and hurts | Paired correctness counts and six reviewed test examples, three per effect | Completed in the paired results |

The [data loader](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/blob/df106744b01854ce2b70065872ae6341b8b36b28/artifacts/discourse_data.py#L17) matches the baseline filtering: synthetic placeholders, missing labels, blank cleaned text, and `other` are excluded. Explicit deleted/removed markers remain eligible. Parent text is metadata, so enabling context does not alter example eligibility. Roots and synthetic-parent cases use empty parent text; parent labels are not inputs.

The existing seed-42 conversation hash split is reused, with 79,517 training, 9,373 development, and 10,343 test examples. The recovered exports have identical full split manifests. Every prediction was matched to the expected example, conversation, label, and text fingerprint. Recomputed metrics and confusion matrices agree with the saved metrics.

## Results and interpretation

| Metric | Comment only | Parent context | Delta |
|---|---:|---:|---:|
| Development macro F1 | 0.5012 | 0.5785 | +0.0773 |
| Test macro F1 | 0.4996 | 0.5766 | +0.0770 |
| Test accuracy | 0.6428 | 0.7376 | +0.0948 |
| Test weighted F1 | 0.6290 | 0.7387 | +0.1097 |

Seven of nine classes improve on test. Elaboration has the largest F1 gain (+0.2412), followed by disagreement (+0.1292), humor (+0.1167), and answer (+0.1148). Announcement (-0.0041) and negative reaction (-0.0029) decline. Full-precision values and support are in the [test per-class CSV](parent-context/test-per-class-f1.csv); the [development CSV](parent-context/dev-per-class-f1.csv) provides a cross-check.

The context model reduces elaboration-to-answer confusion from 51.1% to 14.0%, and disagreement-to-answer confusion from 40.1% to 8.6%. The reviewed examples also show regressions when question versus disagreement, agreement versus humor, or disagreement versus negative reaction overlap. These are changes in correctness against corpus labels, not causal explanations of the model's internal reasoning.

The [context encoder](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/blob/df106744b01854ce2b70065872ae6341b8b36b28/artifacts/discourse_transformer.py#L29) reserves half the pair token budget for each input and decodes then retokenizes them. Comment-only encoding uses a single input with the full length limit. The comparison therefore changes reply truncation and processing along with adding parent text. It uses one training seed; no significance or cross-community generalization claim is made.

The analysis preserves empty-parent cases and deletion markers to reproduce the published scores. Among the 1,409 empty-parent test examples, the two configurations still differ by 19 net correct predictions. This further limits attributing every gain directly to parent information. Detailed availability strata and the remaining limitations are in the [results report](parent-context/README.md).

## Reproduce the completed analysis

The original exports were recovered from Jay's shared transformer outputs. Each of `plain_test`, `parent_context_test`, `plain_dev`, and `parent_context_dev` needs `evaluation_config.json`, `split_manifest.json`, `transformer/metrics.json`, and `transformer/predictions.jsonl`. Full exports stay under ignored `data/local/`; model checkpoints are not needed.

For example, after placing the exports under `data/local/jay-transformer-exports/experiment_01/`, run from the repository root with an empty output directory:

```bash
uv run python artifacts/compare_parent_context.py \
  --plain-evaluation data/local/jay-transformer-exports/experiment_01/plain_test \
  --context-evaluation data/local/jay-transformer-exports/experiment_01/parent_context_test \
  --output data/local/parent-context-reproduction/test
```

The command audits equal split metadata and manifests, conversation isolation, complete populations, unique IDs, source text, labels, and stored metrics. Pairing is by ID and independent of file order. It preserves an existing nonempty output directory and produces no report when an input audit fails.

It writes aggregate `summary.json`, `per_class_f1.csv`, and a report with its default reproducible example sample. The six short examples in the completed report are a separate manual selection, retained with exact IDs and text in [examples.json](parent-context/examples.json). Both split commands and runtime versions are documented in the full report.

## Validation

The merged modeling and comparison code passed 107 automated tests when checked together. The comparison tests use synthetic fixtures for alignment, fingerprints, split isolation, stale metrics, all four correctness transitions, and output preservation. This follow-up additionally runs the audit on the original 9,373 development and 10,343 test predictions and verifies the committed aggregates and selected examples against those exports.

Local checks used the existing `.venv/bin/python` because `uv` was unavailable. The original GPU training was not rerun.
