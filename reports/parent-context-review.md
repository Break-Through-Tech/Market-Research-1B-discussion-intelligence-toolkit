# October Task 3.5 review and paired analysis

PR #21 already implements the core parent-comment context experiment. This follow-up completes the review, comparison command, and regression tests. Comparing per-class results side by side and inspecting the same examples under both models is deferred to issue #22 until the original evaluation exports are available.

**Status: review and comparison tooling complete; empirical paired analysis deferred to [issue #22](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/issues/22).** The exports and trained checkpoints are absent from this checkout and the accessible team Drive. Aggregate notebook scores cannot reconstruct per-example predictions. The review and tested tooling are ready to merge; issue #22 remains open for the missing results and example review.

## Review of PR 21

Reviewed on November 1, 2026, at commit [`df106744b01854ce2b70065872ae6341b8b36b28`](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/pull/21/commits/df106744b01854ce2b70065872ae6341b8b36b28). Modeling and reported training results belong to that PR; this follow-up reviews and analyzes its outputs.

| Task 3.5 requirement | Evidence in PR #21 | Remaining work |
|---|---|---|
| Construct parent-reply inputs | `load_discourse_data` resolves the immediate parent within each standardized conversation | Covered |
| Compare comment-only and parent-context classifiers | Separate three-epoch DistilBERT experiments use the same examples and conversation-level split | Covered |
| Report macro F1 | Development and test comparisons are published in the notebook | Covered |
| Compare per-class F1 | The published test table shows only the selected context model | Add comment-only F1, context F1, deltas, and support in one table |
| Show examples where context helps and hurts | The notebook displays errors from the selected model and class-level confusion comparisons | Pair both models' predictions by example ID and inspect both correctness transitions |

The [data loader](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/blob/df106744b01854ce2b70065872ae6341b8b36b28/artifacts/discourse_data.py#L17) applies the baseline notebook's filtering: synthetic placeholders, missing labels, blank cleaned text, and `other` are excluded. Explicit deleted/removed markers remain eligible. Parent text is metadata, so enabling context does not alter example eligibility; roots and synthetic-parent cases use an empty parent string. Parent labels are not model inputs.

The existing seed-42 conversation hash split is reused. The notebook reports 79,517 training, 9,373 development, and 10,343 test examples. Split manifests fingerprint the reply text, gold label, and parent text for every example. The [evaluation script](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/blob/df106744b01854ce2b70065872ae6341b8b36b28/scripts/evaluate_discourse_models.py#L22) exports predictions with example and conversation IDs, full reply/parent text, gold labels, and predicted labels. That is sufficient for the remaining paired analysis; training need not be repeated if the exports are preserved.

All **98 tests** in PR #21 passed in an isolated local checkout, including its tiny-checkpoint CPU integration tests. The reported full-corpus GPU experiments were not rerun during this review.

## Published results

These are results reported by [PR #21's notebook](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/blob/df106744b01854ce2b70065872ae6341b8b36b28/notebooks/discourse_transformer_colab.ipynb), rather than new measurements from this follow-up.

| Metric | Comment only | Parent context | Delta |
|---|---:|---:|---:|
| Development macro F1 | 0.5012 | 0.5785 | +0.0773 |
| Test macro F1 | 0.4996 | 0.5766 | +0.0770 |
| Test accuracy | 0.6428 | 0.7376 | +0.0948 |
| Test weighted F1 | 0.6290 | 0.7387 | +0.1097 |

The selected parent-context model's published test F1 is 0.6187 for agreement, 0.4348 for announcement, 0.8378 for answer, 0.7896 for appreciation, 0.4522 for disagreement, 0.6313 for elaboration, 0.2957 for humor, 0.2987 for negative reaction, and 0.8303 for question. The corresponding comment-only values are absent from the published table, so per-class deltas are not filled in here.

The notebook reports a development F1 decline for negative reaction from 0.3370 to 0.3117 with context. This establishes that gains are not uniform across classes; it does not identify particular comments whose predictions became wrong. A sample of the context model's errors alone also cannot tell whether the comment-only model got those examples right.

## Interpretation limits

The [context encoder](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/blob/df106744b01854ce2b70065872ae6341b8b36b28/artifacts/discourse_transformer.py#L29) reserves half the pair token budget for each input and decodes then retokenizes them. The comment-only encoder uses a single input with the full length limit. The reported improvement therefore compares practical input configurations; it does not isolate parent information from the changed reply budget and encoding path. The upstream notebook acknowledges this limitation.

The analysis should preserve the original example population, including empty-parent cases and deletion markers, to reproduce the published scores. It additionally reports correctness changes and accuracy separately for nonempty and empty parent text. Nonempty text can still be a deletion marker; these strata describe exported text availability rather than guaranteed semantic context.

The experiments use one training seed. No claim of statistical significance or cross-community generalization follows from this comparison. "Helps" and "hurts" describe correctness against corpus labels, which can be ambiguous; illustrative snippets are not causal explanations of the model's behavior.

## Deferred paired analysis

The missing artifacts are the two evaluation directories from the original run:

```text
data/local/discourse_transformer/experiment_01/
  plain_test/
    evaluation_config.json
    split_manifest.json
    transformer/metrics.json
    transformer/predictions.jsonl
  parent_context_test/
    evaluation_config.json
    split_manifest.json
    transformer/metrics.json
    transformer/predictions.jsonl
```

To complete issue #22, recover both directories from the original run. If only saved checkpoints survive, PR #21's evaluation script can regenerate the exports by inference. If neither exports nor checkpoints survive, reproducing the original paired analysis requires another training run and should be tracked separately. The committed notebook outputs contain neither complete predictions nor model weights. This artifact recovery and empirical analysis are deferred follow-up work and do not block merging the review and comparison tooling.

Once the directories are available, run from the repository root:

```bash
uv run python artifacts/compare_parent_context.py \
  --plain-evaluation data/local/discourse_transformer/experiment_01/plain_test \
  --context-evaluation data/local/discourse_transformer/experiment_01/parent_context_test \
  --output data/local/parent-context-comparison
```

The command checks equal split metadata and manifests, conversation isolation, complete example populations, unique IDs, matching labels and source text, and text fingerprints. It independently recomputes the saved metrics before comparing the models. Pairing is by ID and is independent of file row order. A failed audit produces no report. An existing nonempty output directory is preserved.

It writes `summary.json` with both models' metrics and input SHA-256 fingerprints, `per_class_f1.csv` with support and F1 deltas, and `report.md` with aggregate scores, correctness transitions, parent-availability strata, and up to three examples per effect. Example selection is reproducible and varies gold classes using sampling seed 42; the sampling seed does not alter the evaluation split. Snippets are capped at 400 characters and link to Reddit when canonical IDs permit. The corpus exports are the evidence if live Reddit pages have changed.

Full predictions stay under ignored `data/local/`. In a follow-up PR for issue #22, review the generated report and add the aggregate results and selected examples. Complete #22 only after the missing comparisons have been verified. No paired metrics or example-level findings are claimed while the exports are unavailable.

## Validation

The new tests use synthetic export fixtures to exercise all four correctness transitions, ordering differences, parent availability, mismatched IDs and fingerprints, split leakage, stale metrics, and output preservation. They do not download models or claim full-corpus results.

```bash
uv run python -m unittest discover -s tests -v
```

Local checks used the existing `.venv/bin/python` in place of `uv run python` because the `uv` executable was unavailable. The comparison tool depends only on the repository's existing evaluation stack and can be merged independently of PR #21's model modules.
