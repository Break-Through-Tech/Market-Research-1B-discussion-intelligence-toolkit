# Parent-context paired evaluation results

The original evaluation exports supplied by Jay reproduce PR #21’s published results. On the same 10,343 held-out test comments, adding parent context raises macro F1 from **0.4996 to 0.5766**. It corrects **1,849** comment-only errors while introducing **868** errors, a net gain of **981** correct predictions. The largest per-class improvement is for elaboration. Announcement and negative reaction decline slightly.

This analysis completes the remaining evaluation work for [October task 3.5, issue #22](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/issues/22). The models and their original training are from [PR #21](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/pull/21); the comparison tool is from [PR #23](https://github.com/Break-Through-Tech/Market-Research-1B-discussion-intelligence-toolkit/pull/23). Analysis completed November 1, 2026, by reusing saved predictions. No training or inference was repeated.

## Scope and artifact checks

The two models use the same baseline-compatible Coarse Discourse examples: synthetic placeholders, missing labels, blank cleaned text, and `other` are excluded. Deleted/removed markers and roots remain eligible. Parent text is empty for roots and unavailable or synthetic parents. Parent labels are not inputs. The [implementation review](../parent-context-review.md) explains the original data and modeling decisions.

The comparison command verified all of the following before calculating results:

- The four export directories use an identical full split manifest and seed 42, with no conversation crossing train, development, and test splits.
- Every expected evaluation example occurs exactly once, with matching IDs, conversation IDs, gold labels, reply text, and parent text.
- Text and label fingerprints agree with the manifests.
- Recomputed accuracy, macro F1, weighted F1, per-class F1, and confusion matrices agree with the saved metrics.

The [test summary](test-summary.json) and [development summary](dev-summary.json) contain the recomputed metrics, confusion matrices, paired counts, and SHA-256 hashes of every analysis input. Full predictions, manifests, and checkpoints remain outside Git.

## Overall comparison

| Split | Examples | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---|---:|---:|---:|
| dev | 9,373 | Comment only | 0.6514 | 0.5012 | 0.6386 |
| dev | 9,373 | Parent context | 0.7362 | 0.5785 | 0.7365 |
| test | 10,343 | Comment only | 0.6428 | 0.4996 | 0.6290 |
| test | 10,343 | Parent context | 0.7376 | 0.5766 | 0.7387 |

The test macro-F1 gain is **0.0770**, or 7.70 percentage points. Accuracy rises by **9.48 percentage points**. The development macro-F1 gain is 0.0773. These are descriptive differences for the saved runs, with no significance claim.

## Per-class test F1

[Full-precision CSV](test-per-class-f1.csv). Support is the number of test examples with that gold label. Delta is parent context minus comment only.

| Class | Support | Comment only | Parent context | Delta |
|---|---:|---:|---:|---:|
| agreement | 516 | 0.5573 | 0.6187 | +0.0614 |
| announcement | 134 | 0.4388 | 0.4348 | -0.0041 |
| answer | 4,304 | 0.7231 | 0.8378 | +0.1148 |
| appreciation | 936 | 0.7689 | 0.7896 | +0.0208 |
| disagreement | 409 | 0.3229 | 0.4522 | +0.1292 |
| elaboration | 2,004 | 0.3901 | 0.6313 | +0.2412 |
| humor | 251 | 0.1790 | 0.2957 | +0.1167 |
| negativereaction | 173 | 0.3016 | 0.2987 | -0.0029 |
| question | 1,616 | 0.8146 | 0.8303 | +0.0157 |

Seven of nine classes improve on test. Elaboration has the largest gain (+0.2412), followed by disagreement (+0.1292), humor (+0.1167), and answer (+0.1148). Announcement (-0.0041) and negative reaction (-0.0029) decline. Humor and negative reaction remain difficult even after context is added.

The test confusion matrices show fewer continuations or counterarguments being labeled as ordinary answers:

| Gold class predicted as answer | Comment only | Parent context |
|---|---:|---:|
| Elaboration | 1,025 / 2,004 (51.1%) | 281 / 2,004 (14.0%) |
| Disagreement | 164 / 409 (40.1%) | 35 / 409 (8.6%) |

## Development cross-check

[Full-precision CSV](dev-per-class-f1.csv). These development results complement the held-out test comparison. They are not a new tuning round.

| Class | Support | Comment only | Parent context | Delta |
|---|---:|---:|---:|---:|
| agreement | 494 | 0.5557 | 0.5973 | +0.0416 |
| announcement | 146 | 0.3782 | 0.5500 | +0.1718 |
| answer | 3,979 | 0.7289 | 0.8387 | +0.1098 |
| appreciation | 827 | 0.7472 | 0.7672 | +0.0200 |
| disagreement | 322 | 0.3497 | 0.4163 | +0.0665 |
| elaboration | 1,665 | 0.3727 | 0.5978 | +0.2251 |
| humor | 203 | 0.2058 | 0.2850 | +0.0792 |
| negativereaction | 178 | 0.3370 | 0.3117 | -0.0253 |
| question | 1,559 | 0.8356 | 0.8429 | +0.0073 |

Elaboration improves strongly on both splits, and negative reaction declines on both. Announcement improves on development but declines slightly on test, so its development gain should not be treated as a reliable class-level benefit.

## Paired correctness changes

Helps means the comment-only prediction is wrong and the context prediction is correct. Hurts means the reverse. Both-wrong includes cases where the predicted label changes but remains incorrect.

| Split | Helps | Hurts | Both correct | Both wrong | Net additional correct |
|---|---:|---:|---:|---:|---:|
| dev | 1,577 | 783 | 5,323 | 1,690 | +794 |
| test | 1,849 | 868 | 5,780 | 1,846 | +981 |

### Parent-text availability on test

Nonempty parent text includes 244 `[deleted]` or `[removed]` markers. It describes text availability, not necessarily useful semantic context.

| Parent text | Examples | Helps | Hurts | Comment-only accuracy | Context accuracy |
|---|---:|---:|---:|---:|---:|
| Nonempty | 8,934 | 1,705 | 743 | 0.6352 | 0.7429 |
| Empty | 1,409 | 144 | 125 | 0.6906 | 0.7040 |

Most of the net accuracy gain occurs among comments with nonempty parent text: 962 additional correct predictions, compared with 19 among empty-parent examples. Changes on empty-parent examples also show why this is a comparison of two trained configurations rather than a causal measure of the parent’s words. The models have different learned weights and input processing.

## Reviewed examples

The following six test examples were manually selected for short, interpretable examples across several classes. They are illustrative, not a random or representative sample. Gold labels come from the corpus. The interpretations describe the text and prediction changes, without claiming to explain the model’s internal reasoning. [Exact text and IDs](examples.json) are retained for auditing; live Reddit pages may have changed.

### 1. Context helps: elaboration

[Source comment](https://www.reddit.com/comments/bqf0l/_/c0o2iu2/) | `reddit:reddit-coarse-discourse-corpus:utterance:t1_c0o2iu2`

Gold: **elaboration**. Comment only: **answer**. Parent context: **elaboration**.

Parent:

    Might not work very so well considering the low light.

Reply:

    A good HD camera could output 30fps but shoot with 1/4 second exposures. Regular video looks funny, but a nice stationary sky might look pretty good.

The reply develops the parent’s point about low light. The context model matches the gold elaboration label, while the comment-only model treats it as an answer.

### 2. Context helps: disagreement

[Source comment](https://www.reddit.com/comments/32xel2/_/cqfrm9j/) | `reddit:reddit-coarse-discourse-corpus:utterance:t1_cqfrm9j`

Gold: **disagreement**. Comment only: **answer**. Parent context: **disagreement**.

Parent:

    I think he's an anomaly.

Reply:

    Nah, he's falling back to earth

The reply rejects the parent’s claim. The context model matches disagreement rather than treating the reply as an answer.

### 3. Context helps: humor

[Source comment](https://www.reddit.com/comments/16cc39/_/c7uplys/) | `reddit:reddit-coarse-discourse-corpus:utterance:t1_c7uplys`

Gold: **humor**. Comment only: **answer**. Parent context: **humor**.

Parent:

    MY DAD CAN BEAT UP YOUR DAD.

Reply:

    MY DAD CAN BEAT UP HIS DAD WHO CAN BEAT UP YOUR DAD SO BY THE TRANSITIVE PROPERTY MY DAD CAN BEAT UP YOURS!

The exaggerated reply continues the parent’s joke. The context model matches humor rather than an ordinary answer.

### 4. Context hurts: agreement

[Source comment](https://www.reddit.com/comments/25ie0d/_/chhq4fp/) | `reddit:reddit-coarse-discourse-corpus:utterance:t1_chhq4fp`

Gold: **agreement**. Comment only: **agreement**. Parent context: **humor**.

Parent:

    Yer a wizard, 'Arry!

Reply:

    It all makes sense.

A joking parent is followed by a reply labeled agreement. The context model predicts humor. This illustrates overlap between tone and the reply’s discourse role.

### 5. Context hurts: question

[Source comment](https://www.reddit.com/comments/sgi18/_/c4dua8u/) | `reddit:reddit-coarse-discourse-corpus:utterance:t1_c4dua8u`

Gold: **question**. Comment only: **question**. Parent context: **disagreement**.

Parent:

    you're joking, right? nickelback doesn't have talent

    (sorry, someone was going to bash nickelback anyways, I may has well)

Reply:

    I don't like them, but for a band to make it big time, they must have at least some musical talent, right?

The reply is phrased as a question while challenging the parent’s claim. The context model predicts disagreement instead of the gold question label, illustrating overlapping discourse functions.

### 6. Context hurts: disagreement

[Source comment](https://www.reddit.com/comments/4g4hxw/_/d2f7sb8/) | `reddit:reddit-coarse-discourse-corpus:utterance:t1_d2f7sb8`

Gold: **disagreement**. Comment only: **disagreement**. Parent context: **negativereaction**.

Parent:

    On pc... ctrl + prnt scn, ctrl + v (paste)in paint.. Ps4 has a share button, xb1 has a share option as well.

Reply:

    That's a stupid solution for PC, since you have screenshot button on steam which is F12 (by default). With your way you can take only one screenshot instead of few.

The reply criticizes the proposed screenshot method and offers an alternative. The context model predicts negative reaction instead of disagreement; negative tone and a counterargument can occur together.

## Interpretation limits

The results support the existing selection of the parent-context configuration for this dataset. They also show a substantial number of regressions, overlapping discourse functions, and uneven per-class gains.

The context encoder reserves half the pair token budget for each input and decodes then retokenizes them; comment-only encoding uses a single input with the full length budget. Thus the comparison changes reply truncation and processing along with adding parent text. One training seed was used, and no uncertainty intervals or cross-community validation are claimed. Ambiguous gold labels and deleted text further limit interpretation.

## Reproduce

Recover the four evaluation directories from the team-shared `experiment_01` outputs and place them under `data/local/jay-transformer-exports/experiment_01/`. Each directory needs `evaluation_config.json`, `split_manifest.json`, `transformer/metrics.json`, and `transformer/predictions.jsonl`. No model checkpoint is required for this analysis. Use empty output directories:

```bash
uv run python artifacts/compare_parent_context.py \
  --plain-evaluation data/local/jay-transformer-exports/experiment_01/plain_test \
  --context-evaluation data/local/jay-transformer-exports/experiment_01/parent_context_test \
  --output data/local/parent-context-reproduction/test

uv run python artifacts/compare_parent_context.py \
  --plain-evaluation data/local/jay-transformer-exports/experiment_01/plain_dev \
  --context-evaluation data/local/jay-transformer-exports/experiment_01/parent_context_dev \
  --output data/local/parent-context-reproduction/dev
```

The command reproduces the aggregate summaries and per-class CSVs, and creates its default seed-42 example sample. The six examples above are a separate manual selection, identified exactly in `examples.json`; the default sample is not claimed to reproduce that selection.

The recorded recomputation used the existing `.venv/bin/python` because `uv` was unavailable. Runtime versions: Python 3.12.11, scikit-learn 1.9.1, numpy 2.5.3, scipy 1.18.1, pydantic 2.13.5. The original training and checkpoints were not modified.
