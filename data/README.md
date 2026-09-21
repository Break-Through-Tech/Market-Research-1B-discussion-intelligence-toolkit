# Data

This directory implements the project's two-tier data-access model:

1. A small, version-controlled reference sample for inspection, schema work, and smoke tests.
2. Reproducible download instructions for the complete source corpora, which are not committed to this repository.

## Tier 1 — tracked reference data

`coarse-discourse-reference-sample.jsonl` contains two complete, source-derived Reddit conversations from ConvoKit's [Coarse Discourse Sequence Corpus](https://convokit.cornell.edu/documentation/coarseDiscourse.html):

| Conversation ID | Community | Utterances | Purpose |
|---|---:|---:|---|
| `t3_10r2f8` | `buildapc` | 7 | Technical-help discussion |
| `t3_169c79` | `explainlikeimfive` | 4 | Explanatory discussion |

The sample excludes Reddit usernames. Each JSONL record has these fields:

| Field | Meaning |
|---|---|
| `source` | Source corpus identifier |
| `conversation_id` | Reddit root-post ID |
| `subreddit` | Source community |
| `url` | Original Reddit discussion URL |
| `title` | Root-post title |
| `utterances` | Complete conversation in source order |
| `utterances[].id` | Reddit post or comment ID |
| `utterances[].parent_id` | Immediate parent ID; `null` for the root post |
| `utterances[].depth` | Source-recorded reply depth |
| `utterances[].text` | Source utterance text |
| `utterances[].discourse_act` | Coarse Discourse majority label |
| `utterances[].score` | Source vote score; `null` when unavailable |

The file is a source-normalized reference fixture, not the toolkit's final canonical schema: the upstream corpus does not provide timestamps for these selected records.

## Tier 2 — complete source corpora

Use the team's shared Python 3.12 setup. From the repository root, install the
locked dependencies once with `uv sync --locked`, then run the download scripts
below with `uv run`. Dependencies are managed in `pyproject.toml` and `uv.lock`.

### Coarse Discourse Sequence Corpus

Use this corpus for the initial threaded-discussion path and discourse-act classification. It contains 9,483 Reddit conversations and 115,827 utterances with reply relationships and discourse labels.

```text
uv run python data/Download_Discourse_Corpus.py
```

Source and documentation: <https://convokit.cornell.edu/documentation/coarseDiscourse.html>

### Conversations Gone Awry — ChangeMyView

Use this corpus for conversation-quality and derailment modeling. The prepared loader names it `conversations-gone-awry-cmv-corpus`.

```text
uv run python data/Download_Awry_Corpus.py
```

Source documentation: <https://convokit.cornell.edu/documentation/>

### Deferred Pushshift bulk-corpus candidate

The prepared starter notebook identifies these static Hugging Face datasets:

- <https://huggingface.co/datasets/fddemarco/pushshift-reddit>
- <https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments>

They are not approved as input to the current `RedditDumpConnector`. On 2026-08-18, the published comments schema exposed `link_id` but not `parent_id`; the connector requires `parent_id` to reconstruct nested replies. Select a parent-preserving export and validate its schema before using it for end-to-end thread analysis.

## Repository and source constraints

Do not commit large corpora, private data, API keys, model checkpoints, or generated local parquet files. Record the source URL, retrieval date, selected communities, retained-record counts, split strategy, and relevant source terms for every corpus used beyond this reference sample. The upstream source's terms govern corpus access and redistribution.

## Standardized ConvoKit outputs (Task #2)

Both complete corpora have been standardized without dropping source records.
See [STANDARDIZATION.md](STANDARDIZATION.md) for exact counts, cleaning rules,
schema 1.1 changes, validation, reproduction commands, and the Drive handoff.
Machine-readable counts are in [standardization-summary.json](standardization-summary.json).
Small canonical examples are in `standardized-samples/`; full files are generated
in Git-ignored `data/local/standardized-v1/` and belong on the shared Drive.
These canonical examples differ from the original Tier 1 reference fixture above.
