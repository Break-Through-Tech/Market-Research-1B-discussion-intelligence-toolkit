"""Sentence embeddings for the standardized corpora, with an on-disk cache.

One embedding per real utterance. Synthetic placeholder utterances are excluded:
they were inserted during standardization to repair missing-parent references,
have empty text, and are not real messages.

Empty and deleted utterances are kept so row counts line up with the text
features table, but they are flagged (`is_empty`, `is_deleted`) so model
comparisons and clustering can filter them out.

Cache layout, one folder per (model, corpus):

    data/local/embeddings/<model-slug>/<corpus>/
        embeddings.npy   (N, D) array, L2-normalized, row i <-> ids.parquet row i
        ids.parquet      utterance_id, conversation_id, discourse_act, flags
        manifest.json    model, dimension, row count, exclusion rules; written last

`manifest.json` is written after the other two files, so a folder without one
is an incomplete run and `load_cache` refuses it.

Usage:

    uv run python artifacts/embeddings.py --model all-MiniLM-L6-v2
    uv run python artifacts/embeddings.py --model all-mpnet-base-v2 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from corpora import _bucket_for
from schema import Conversation, Utterance
from standardize_convokit import iter_standardized

# Candidates for the comparison. Dimensions are listed so results tables and
# disk estimates do not need the model loaded.
CANDIDATE_MODELS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
}

STANDARDIZED_DIR = Path("data/local/standardized-v1")
CORPORA = {
    "coarse-discourse": STANDARDIZED_DIR / "reddit-coarse-discourse-corpus.jsonl",
    "cga-cmv": STANDARDIZED_DIR / "conversations-gone-awry-cmv-corpus.jsonl",
}
DEFAULT_CACHE_ROOT = Path("data/local/embeddings")

CACHE_FORMAT_VERSION = 1
ID_COLUMNS = ("utterance_id", "conversation_id", "discourse_act", "is_deleted", "is_empty")


# ---------------------------------------------------------------------------
# Building the text table
# ---------------------------------------------------------------------------


def _is_synthetic(utterance: Utterance) -> bool:
    return bool(utterance.metadata.get("is_synthetic", False))


def build_text_table(conversations: Iterable[Conversation], limit: int | None = None) -> pd.DataFrame:
    """One row per real utterance, in corpus order, with the text to embed.

    `limit` caps the number of rows, which is useful for quick comparisons on
    CPU. Rows are taken in corpus order, so whole conversations stay together
    (apart from the last one, which may be cut).
    """
    rows = []
    for conversation in conversations:
        for u in conversation.utterances:
            if _is_synthetic(u):
                continue
            text = u.text_clean or ""
            rows.append({
                "utterance_id": u.id,
                "conversation_id": u.conversation_id,
                "discourse_act": u.discourse_act,
                "is_deleted": u.is_deleted,
                "is_empty": not text.strip(),
                "text": text,
            })
            if limit is not None and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=[*ID_COLUMNS, "text"])


def load_text_table(corpus: str, limit: int | None = None) -> pd.DataFrame:
    """`build_text_table` for one of the named corpora in `CORPORA`."""
    return build_text_table(iter_standardized(CORPORA[corpus]), limit=limit)


def sample_conversations(
    conversations: Iterable[Conversation],
    fraction: float,
    key: Callable[[Conversation], str] = lambda c: c.id,
    seed: int = 42,
) -> Iterator[Conversation]:
    """Keep a deterministic ~`fraction` of conversations, chosen by hashing `key`.

    Same hashing as `corpora.split_by_group`, so a sample is reproducible and
    spread across the whole corpus rather than taken from the start. Pass a
    different `key` to keep groups together, e.g. CGA-CMV pairs by `pair_id`.
    """
    for conversation in conversations:
        if _bucket_for(key(conversation), seed) < fraction:
            yield conversation


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """`TextEmbedder` (see evaluation.py) backed by a sentence-transformers model.

    The model is loaded lazily on first use, so constructing one is cheap.
    Embeddings are L2-normalized: cosine similarity is a plain dot product.
    """

    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def embed(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return vectors.astype(np.float32, copy=False)

    def count_truncated(self, texts: Sequence[str]) -> int:
        """Number of texts longer than the model's token limit (cut off when encoded)."""
        lengths = self.model.tokenizer(list(texts), add_special_tokens=True)["input_ids"]
        return sum(len(ids) > self.max_seq_length for ids in lengths)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def model_slug(model_name: str) -> str:
    """Folder-safe model name: 'BAAI/bge-small-en-v1.5' -> 'BAAI__bge-small-en-v1.5'."""
    return model_name.replace("/", "__")


def cache_dir(model_name: str, corpus: str, root: Path = DEFAULT_CACHE_ROOT) -> Path:
    return Path(root) / model_slug(model_name) / corpus


@dataclass
class CachedEmbeddings:
    ids: pd.DataFrame
    embeddings: np.ndarray
    manifest: dict


def save_cache(
    directory: Path,
    table: pd.DataFrame,
    embeddings: np.ndarray,
    manifest: dict,
    dtype: str = "float32",
) -> Path:
    """Write ids, vectors and manifest. The manifest goes last to mark completion."""
    if len(table) != len(embeddings):
        raise ValueError(f"{len(table)} rows but {len(embeddings)} embeddings")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest_path = directory / "manifest.json"
    # Remove a stale manifest first, so a crash mid-write cannot leave an old
    # manifest describing new files.
    manifest_path.unlink(missing_ok=True)

    table.loc[:, list(ID_COLUMNS)].to_parquet(directory / "ids.parquet", index=False)
    np.save(directory / "embeddings.npy", embeddings.astype(dtype, copy=False))

    manifest = {
        **manifest,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "n_rows": int(len(table)),
        "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "dtype": dtype,
        "normalized": True,
        "excluded": "synthetic placeholder utterances (metadata.is_synthetic)",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def load_cache(directory: Path, mmap: bool = False) -> CachedEmbeddings:
    """Load a cache folder and check that ids, vectors and manifest agree.

    `mmap=True` memory-maps the vectors instead of reading them into RAM.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{directory}: no manifest.json (missing or incomplete run)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = pd.read_parquet(directory / "ids.parquet")
    embeddings = np.load(directory / "embeddings.npy", mmap_mode="r" if mmap else None)

    if not (len(ids) == len(embeddings) == manifest["n_rows"]):
        raise ValueError(
            f"{directory}: row mismatch (ids={len(ids)}, "
            f"embeddings={len(embeddings)}, manifest={manifest['n_rows']})"
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != manifest["dimension"]:
        raise ValueError(f"{directory}: expected dimension {manifest['dimension']}, got {embeddings.shape}")
    return CachedEmbeddings(ids=ids, embeddings=embeddings, manifest=manifest)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def embed_and_cache(
    corpus: str,
    embedder: SentenceTransformerEmbedder,
    root: Path = DEFAULT_CACHE_ROOT,
    limit: int | None = None,
    dtype: str = "float32",
    table: pd.DataFrame | None = None,
    extra_manifest: dict | None = None,
) -> Path:
    """Embed every real utterance of `corpus` and write the cache folder.

    Pass `table` to reuse an already-built text table across models (for
    example a sample), and `extra_manifest` to record how it was built.
    """
    if table is None:
        table = load_text_table(corpus, limit=limit)

    embedder.model  # load before timing, so encode_seconds is encoding only
    start = time.perf_counter()
    vectors = embedder.embed(table["text"].tolist(), show_progress_bar=True)
    seconds = time.perf_counter() - start

    manifest = {
        "model_name": embedder.model_name,
        "corpus": corpus,
        "source_path": str(CORPORA.get(corpus, "")),
        "limit": limit,
        "max_seq_length": embedder.max_seq_length,
        "encode_seconds": round(seconds, 1),
        "sentence_transformers_version": version("sentence-transformers"),
        "n_empty": int(table["is_empty"].sum()),
        "n_deleted": int(table["is_deleted"].sum()),
        **(extra_manifest or {}),
    }
    return save_cache(cache_dir(embedder.model_name, corpus, root), table, vectors, manifest, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="sentence-transformers model name")
    parser.add_argument("--corpus", choices=[*CORPORA, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N real utterances")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()

    if args.limit is not None and args.cache_root == DEFAULT_CACHE_ROOT:
        # Keep sample runs from overwriting the full cache for the same model.
        args.cache_root = DEFAULT_CACHE_ROOT / f"sample-{args.limit}"

    embedder = SentenceTransformerEmbedder(args.model, batch_size=args.batch_size)
    corpora = list(CORPORA) if args.corpus == "all" else [args.corpus]
    for corpus in corpora:
        out = embed_and_cache(corpus, embedder, root=args.cache_root, limit=args.limit, dtype=args.dtype)
        cached = load_cache(out, mmap=True)
        print(f"{corpus}: {cached.manifest['n_rows']} x {cached.manifest['dimension']} "
              f"in {cached.manifest['encode_seconds']}s -> {out}")


if __name__ == "__main__":
    main()
