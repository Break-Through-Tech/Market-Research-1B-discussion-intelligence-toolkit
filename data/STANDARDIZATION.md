# Task #2 — standardized ConvoKit handoff

This pipeline processes the complete Coarse Discourse and CGA-CMV corpora directly
from their ConvoKit export folders. It does not use the Pushshift-specific
`RedditDumpConnector`, select new sources, train models, or alter the raw downloads.

## Results (2026-09-19)

| Measure | Coarse Discourse | CGA-CMV |
|---|---:|---:|
| Source conversations retained | 9,483 | 6,842 |
| Source utterances retained | 115,827 | 42,964 |
| Unique source speaker IDs used | 63,573 | 9,548 |
| Added missing-parent placeholders | 188 | 0 |
| Output utterances, including placeholders | 116,015 | 42,964 |
| Unique output speaker IDs, including synthetic speakers | 63,699 | 9,548 |
| Conversations with structural repairs | 126 | 0 |
| Replies referencing an absent parent | 207 | 0 |
| Missing source timestamps | 115,827 | 0 |
| Null source text | 1,318 | 0 |
| Empty cleaned source text (includes null text) | 2,575 | 0 |
| Explicit `[deleted]` / `[removed]` text markers | 1,016 | 1,192 |
| Rejected conversations / utterances | 0 / 0 | 0 / 0 |

Counts overlap: null text, empty text and deleted-text markers are different
conditions. Speaker IDs represent source identifiers, not verified unique people;
`[deleted]` accounts cannot be distinguished. Synthetic speaker IDs are scoped to
individual repaired conversations. No subreddit filtering or sampling was applied.

## Reproduce

Use the team's shared setup from PR #4. From the repository root, `uv` selects
Python 3.12 using `.python-version` and installs the versions in `uv.lock`.
`uv sync --locked` creates or synchronizes `.venv`; `uv run` uses that environment.
Dependencies are managed in `pyproject.toml` and `uv.lock`.

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python artifacts/standardize_convokit.py
uv run python artifacts/verify_standardized.py
```

The scripts default to `~/.convokit/saved-corpora` on your machine. If the raw
corpora are not downloaded yet, use the team's scripts:

```bash
uv run python data/Download_Discourse_Corpus.py
uv run python data/Download_Awry_Corpus.py
```

Then run the standardization and verification commands above. To add or remove
dependencies, use `uv add <package>` or `uv remove <package>` and commit both
`pyproject.toml` and `uv.lock`.

You can point `--source-root` at the parent folder of copies downloaded from Drive.
Keep each entire source folder: `utterances.jsonl` alone does not contain all labels
and metadata. Source SHA-256 values in the manifests allow comparison of local and
Drive copies. Folder names alone do not establish identical contents.

Defaults: full outputs go to `data/local/standardized-v1/` (Git-ignored), and small
fixtures go to `data/standardized-samples/`. Override with `--output-dir` and
`--sample-dir`. A rerun replaces those generated outputs, so use a new output
folder for a separately published version. Source files remain read-only.

The adapter streams source rows into temporary SQLite storage, then processes
one conversation at a time. Conversation/speaker metadata is held in memory.
The independent verification pass retains a compact source-field lookup in memory.

## Output contract and decisions

- One JSONL line is one schema-valid `Conversation`, containing speakers and
  utterances. This differs from ConvoKit's source `utterances.jsonl`, where each
  line is one message. Preserve the original folders separately.
- Schema version **1.1** allows `created_at: null` on conversations and utterances.
  This represents genuinely unknown times; no dates are fabricated. Non-null
  timestamps are UTC. Consumers must handle `None` and should not sort unknown
  timestamps as if they were real dates. Existing dated Reddit objects still work.
- IDs are `reddit:<corpus>:<kind>:<source-id>`; source IDs remain in `source_id`.
  Corpus namespaces prevent collisions when combining datasets. Speaker handles
  and profile metadata are retained in full outputs. The shared schema remains
  source-agnostic; these ID conventions are specific to this adapter.
- Original string content stays unchanged in `text`. Null source text becomes an
  empty string with `metadata.text_was_null=true`; it is never converted to the
  literal string `"None"`. Unexpected non-string/non-null values are rejected.
- `text_clean` decodes HTML entities, normalizes line endings and horizontal
  whitespace, trims lines, and caps consecutive blank lines at one. It preserves
  case, punctuation, emoji, negation, markdown, quote markers, code, and URLs.
  This deliberately avoids deleting conversational evidence; it is not a markdown
  renderer. Whitespace normalization can change indentation, so use raw `text`
  when exact code formatting matters.
- Explicit deleted/removed text markers set `is_deleted=true`. Empty or unknown
  text alone does not prove deletion. A deleted author does not imply deleted
  content. All these source messages stay in the tree and retain their labels.
- Missing parents get one placeholder per missing ID within the conversation.
  The real child's original parent reference is preserved in
  `metadata.source_parent_id`; its canonical parent points to the placeholder.
  Placeholders have empty text, null times, `metadata.is_synthetic=true`, and
  `metadata.reason="missing_parent"`. They attach to the known root solely to
  make traversal possible; `metadata.parent_edge_is_synthetic=true` explicitly
  marks that attachment as invented structure, not recovered ancestry.
- Original source order and recorded depth are preserved in message metadata.
  Output order is depth-first with source-order siblings. Depth is recomputed
  from the standardized tree; 242 source depth values differ in Coarse Discourse.
  In repaired conversations, these depths describe the repaired structure only.
- All discourse labels, including `other`, remain in `discourse_act`; missing
  labels stay null. Full original message metadata is in `metadata.source_metadata`
  except bulky CGA-CMV `parsed` NLP annotations, omitted for all 42,964 messages.
  Each omission is flagged; originals remain available in the raw source folder.
- Conversation labels, `pair_id`, and provided splits are preserved under
  `Conversation.metadata.source_metadata`. Speaker metadata is preserved under
  `Conversation.metadata.source_speaker_metadata`. Existing vectors are retained
  as metadata; no embeddings or quality scores are computed.
- Multiple roots, cycles, invalid references/types and other invalid conversations
  are quarantined with an explanation and source IDs. The command exits with a
  review-required error if any are rejected. Duplicate source message IDs fail
  the import rather than silently overwriting records. This run rejected none.

## How teammates load the files

From the repository root, start Python with `uv run python`, then:

```python
import sys
sys.path.insert(0, "artifacts")
from standardize_convokit import iter_standardized

path = "data/local/standardized-v1/reddit-coarse-discourse-corpus.jsonl"
for conversation in iter_standardized(path):
    root = conversation.root()
    for message in conversation.traverse():
        if message.metadata.get("is_synthetic"):
            continue  # Structural placeholders are not training examples.
        text = message.text_clean
        label = message.discourse_act
        # Perform EDA or feature engineering here.
```

Existing `CoarseDiscourseLoader` / `CgaCmvLoader` consume raw ConvoKit corpora,
not these standardized files. Use `iter_standardized` for this handoff.

## Downstream boundaries

This is a complete standardized source corpus, not a training-ready example set.
Keep synthetic nodes out of text training; for structural features either exclude
repaired conversations or account for synthetic edges explicitly. Report results
on observed edges separately from inferred scaffolding. Handle empty text and
explicit deletion markers deliberately rather than letting them become shortcuts.

CGA-CMV's original splits are retained: train 4,106; val 1,368; test 1,368. Honor
them and preserve pairing/related-thread grouping when selecting examples. Do not
blindly reuse the starter loader's re-splitting or first-four-message heuristic.
For derailment prediction, choose an explicit pre-outcome cutoff: using the whole
thread, removal markers, target metadata or future messages can leak the answer.
`has_removed_comment` is a source outcome label, not a continuous argument-quality
score, and is not copied into `quality_score`.

For Coarse Discourse, split/group at conversation level before deriving message
examples. If creating training subsets, sample whole conversations with a recorded
seed. The team can choose features and training budgets without repeating ingestion.

## Validation and small fixtures

The pipeline validates every exported conversation after JSON serialization:
Pydantic schema, exactly one root, unique message IDs, speaker references, parent
resolution, depths, connectivity and complete traversal. A separate audit checks
source/output checksums, preservation of text, null flags, timestamps, all source
conversation metadata/labels, existing parent edges, and complete record retention.
`source-preservation-check.json` records the full audit result.

Thirteen automated tests cover missing parents, shared placeholders, cycles,
duplicate IDs, multiple roots, null content/timestamps, deleted-author semantics,
label preservation, cleaning, depth/speaker errors, deep trees, source ID namespaces,
and the actual disk pipeline's quarantine/streaming behavior.

The samples contain two complete Coarse Discourse conversations (one repaired)
and one complete CGA-CMV conversation. Speaker identifiers/handles are replaced
with fixture-local participant names, and speaker profile metadata is removed.
Raw message text is preserved, so these are not claimed to be fully anonymized.
They are examples, not representative samples for evaluation.

## GitHub versus Drive

**Commit to GitHub:** the adapter, independent verifier, schema 1.1 adjustment,
tests, `.gitignore`, this guide, `data/README.md`,
`data/standardization-summary.json`, and the tiny `data/standardized-samples/` files.

**Upload to Drive:** the entire `data/local/standardized-v1/` folder, ideally under
`Datasets/standardized/v1/`, alongside the existing untouched raw corpus folders.
It contains two full JSONL files, manifests with source/output SHA-256 values,
validation reports, empty rejection ledgers, and a copy of this guide. Do not
replace the raw datasets or upload `.venv/`. The full outputs are about 265 MiB.

Keep the folder's filenames together so manifest references resolve. Add the
actual Drive folder link to Task #2 when uploaded; no Drive link has been supplied
or verified by this implementation. Upload the folder separately and add its link
to Task #2; the code pull request does not upload the datasets or complete that handoff.

Source documentation:
- https://convokit.cornell.edu/documentation/coarseDiscourse.html
- https://convokit.cornell.edu/documentation/

The upstream source terms govern access and redistribution. Local manifests
record source file checksums and the available original retrieval record; no
additional redistribution rights are asserted by this processing step.
